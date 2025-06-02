import os
import argparse
from typing import Optional

import torch
# Trade precision for performance on Tensor Cores
torch.set_float32_matmul_precision("medium")   
# Help avoid fragmentation on large models 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  
os.environ["NCCL_DEBUG"] = "WARN" 

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader
from torch.optim import AdamW
#  Import the quantization configuration helper 
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,   
    logging,
)
import bitsandbytes as bnb   

logging.set_verbosity_error()

from datasets import Dataset, DatasetDict, Audio
from torchmetrics.text import WordErrorRate   

def load_split(data_root: str, split: str) -> Dataset:
    """
    Load a CSV annotation file for the given split (“train” or “dev”),
    locate each audio file on disk, and return a Hugging Face Dataset
    with the “path” column cast to Audio.
    """
    import pandas as pd

    csv_path = os.path.join(data_root, split, "annotation", f"{split}.csv")
    df = pd.read_csv(csv_path)
    audio_dir = os.path.join(data_root, split, "audio")

    def find_audio(fn: str) -> str:
        """
        Walk the audio directory to find the file named fn.
        """
        for dp, _, files in os.walk(audio_dir):
            if fn in files:
                return os.path.join(dp, fn)
        raise FileNotFoundError(f"{fn} not under {audio_dir}")

    # Replace the CSV’s “path” with the full filesystem path
    df["path"] = df["path"].apply(find_audio)

    # Convert to a Hugging Face Dataset and cast the “path” column to Audio
    ds = Dataset.from_pandas(df, preserve_index=False)
    return ds.cast_column("path", Audio(sampling_rate=16_000))


def prepare_batch(batch, processor: Wav2Vec2Processor):
    """
    Given a dataset batch with:
      - batch["path"]["array"] → raw audio waveform (numpy array)
      - batch["sentence"]        → the transcription text
    Tokenize/process both audio and text, storing:
      - “input_values”   : log-mel features (numpy) for the model
      - “attention_mask” : attention mask (numpy)
      - “labels”         : tokenized transcription (list[int])
    """
    audio = batch["path"]["array"]
    inputs = processor.feature_extractor(
        audio,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_attention_mask=True,
        return_tensors="np",
    )
    batch["input_values"] = inputs.input_values[0]
    batch["attention_mask"] = inputs.attention_mask[0]
    batch["labels"] = processor.tokenizer(
        batch["sentence"],
        add_special_tokens=True,
    ).input_ids
    return batch


class DataCollatorCTCWithPadding:
    """
    Pads a batch of already-extracted “input_values” and “attention_mask” (numpy arrays),
    along with a list of label sequences, into a single PyTorch tensor per field.
    """
    def __init__(self, processor: Wav2Vec2Processor, padding: bool = True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features):
        # Collect lists of numpy arrays or lists:
        input_vals = [f["input_values"] for f in features]
        attention = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        # Pad the audio inputs to max length in this batch
        batch = self.processor.pad(
            {"input_values": input_vals, "attention_mask": attention},
            padding=self.padding,
            return_tensors="pt",
        )

        # Pad the label sequences (token IDs) into a single tensor
        label_batch = self.processor.tokenizer.pad(
            {"input_ids": labels},
            padding=self.padding,
            return_tensors="pt",
        )

        mask = label_batch["attention_mask"]
        # Replace padding tokens in labels with -100 so CTC loss ignores them
        label_ids = label_batch["input_ids"].masked_fill(mask.ne(1), -100)
        batch["labels"] = label_ids
        return batch


class ASRDataModule(L.LightningDataModule):
    """
    LightningDataModule to load, preprocess, and batch the ASR dataset.
    - Uses `datasets.DatasetDict` with splits “train” and “dev”.
    - Maps `prepare_batch` to extract log-mel features and token IDs.
    - Provides PyTorch DataLoaders with dynamic padding.
    """
    def __init__(
        self,
        data_root: str,
        processor: Wav2Vec2Processor,
        batch_size: int = 1,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_root = data_root
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # Called ONCE (on rank 0) to download / preprocess / cache
        self.processor.feature_extractor.sampling_rate = 16_000
        raw = DatasetDict(
            train=load_split(self.data_root, "train"),
            validation=load_split(self.data_root, "dev"),
        )

        # Apply expensive preprocessing exactly once (cached on disk)
        self.dataset = raw.map(
            prepare_batch,
            fn_kwargs={"processor": self.processor},
            remove_columns=["path", "sentence"],
            num_proc=os.cpu_count(),
            load_from_cache_file=True,
        )

    def setup(self, stage: Optional[str] = None):
        # Called on each GPU after prepare_data; can split train/val here if needed
        assert hasattr(self, "dataset"), "Call prepare_data() before setup()"

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            collate_fn=DataCollatorCTCWithPadding(self.processor),
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.batch_size,
            collate_fn=DataCollatorCTCWithPadding(self.processor),
            num_workers=self.num_workers,
        )

class ASRModule(L.LightningModule):
    """
    LightningModule wrapping Wav2Vec2ForCTC. Freezes the feature encoder
    and computes training/validation CTC loss + WER metric via TorchMetrics.
    """
    def __init__(
        self,
        model_name: str,
        learning_rate: float = 3e-4,
        warmup_steps: int = 500,
        total_steps: int = 10000,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load the pretrained processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

        # ─── Quantization configuration via BitsAndBytesConfig ────────────
        qq = BitsAndBytesConfig(
            load_in_4bit=True,              # Enable 4-bit loading 
            bnb_4bit_compute_dtype=torch.float16,  # Compute in float16 
            bnb_4bit_quant_type="nf4",      # NF4 quantization (better accuracy) 
            bnb_4bit_use_double_quant=True   # Use double quant for stability 
        )

        # Load the model using quantization_config + device_map="auto"
        # (Accelerate will shard weights across GPUs automatically) 
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            quantization_config=qq,   # Pass our BitsAndBytesConfig 
        )

        # Freeze the feature encoder to reduce memory usage
        self.model.freeze_feature_encoder()
        # Enable gradient checkpointing on the transformer layers to save memory
        self.model.gradient_checkpointing_enable()

        # Use TorchMetrics’ WER for distributed-friendly metric computation
        self.wer = WordErrorRate()

    def training_step(self, batch, batch_idx):
        out = self.model(**batch)
        self.log("train_loss", out.loss, on_step=True, on_epoch=True, prog_bar=True)
        return out.loss

    def validation_step(self, batch, batch_idx):
        out = self.model(**batch)
        pred_ids = torch.argmax(out.logits, dim=-1)
        pred_str = self.processor.batch_decode(pred_ids)

        labels = batch["labels"].clone()
        # Replace -100 (CTC ignore) with actual pad token ID so decode works
        labels[labels == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(labels, group_tokens=False)

        wer_score = self.wer(pred_str, label_str)

        # sync_dist=True → ensure WER is aggregated correctly under DDP
        self.log("val_wer", wer_score, prog_bar=True, sync_dist=True)
        self.log("val_loss", out.loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # ─── Use the 8-bit Adam optimizer from bitsandbytes ───────────────
        optimizer = bnb.optim.AdamW8bit(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )  # 8-bit Adam has a much smaller memory footprint 

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.total_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

if __name__ == "__main__":
    L.seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model_name", default="facebook/wav2vec2-large-xlsr-53")
    parser.add_argument("--output_dir", default="outputs/asr_run")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # Instantiate the processor (needed both in DataModule and after training)
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)

    # Create DataModule (batch_size=1 by default to reduce peak GPU usage)
    dm = ASRDataModule(
        data_root=args.data_root,
        processor=processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    dm.prepare_data()
    dm.setup()
    train_size = len(dm.dataset["train"])
    total_steps = (train_size // args.batch_size) * args.epochs

    # Instantiate the LightningModule
    model = ASRModule(
        model_name=args.model_name,
        learning_rate=3e-4,
        warmup_steps=500,
        total_steps=total_steps,
    )

    ckpt_cb = ModelCheckpoint(
        monitor="val_wer",
        mode="min",
        filename="asr-{epoch:02d}-{val_wer:.3f}",
        save_weights_only=True,
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=int(os.environ.get("SLURM_NTASKS_PER_NODE", 1)),
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
        strategy=DDPStrategy(find_unused_parameters=True),
        precision="bf16",
        callbacks=[ckpt_cb],
        accumulate_grad_batches=2,
    )

    trainer.fit(model, dm)

    # Save final checkpoint + tokenizer & feature extractor
    trainer.save_checkpoint(os.path.join(args.output_dir, "final.ckpt"))
    processor.save_pretrained(args.output_dir)