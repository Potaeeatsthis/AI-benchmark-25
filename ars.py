import os
import argparse
from typing import Optional
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, Audio, load_metric
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    DataCollatorCTCWithPadding,
    AdamW,
    get_linear_schedule_with_warmup
)

class ASRDataModule(L.LightningDataModule):
    """
    LightningDataModule for ASR data.
    - Reads train/dev CSVs with `path,sentence` columns
    - Casts audio files to HF Audio feature
    - Tokenizes audio and text into model inputs
    """
    def __init__(
        self,
        data_root: str,
        processor: Wav2Vec2Processor,
        batch_size: int = 8,
        num_workers: int = 4,
    ):
        super().__init__()
        # Root directory with 'train 2025-asr_public_dev/' and ' 2025-asr_public_train/' subfolders
        self.data_root = data_root
        # Feature extractor and tokenizer
        self.processor = processor
        # bathch size per GPU
        self.batch_size = batch_size
        # Number of DataLoader workers per GPU
        self.num_workers = num_workers

    def prepare_data(self):
        pass  # assume data already available

    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for training and validation.
        """
        def load_split(root: str, split: str) -> Dataset:
            """
            Read a CSV and return an HF Dataset with audio feature.
            """
            import pandas as pd
            
            # Read CSV with headers `path,sentence`
            csv_path = os.path.join(root, split, "annotation", f"{split}.csv")
            df = pd.read_csv(csv_path)
            
            # Prepend full audio directory to each relative path
            audio_dir = os.path.join(root, split, "audio")
            df["path"] = df["path"].apply(lambda fn: os.path.join(audio_dir, fn))
            
            # Create HF Dataset and cast 'path' column to Audio feature
            ds = Dataset.from_pandas(df, preserve_index=False)
            return ds.cast_column("path", Audio(sampling_rate=16_000))

        # Load both splits
        train_ds = load_split(self.data_root, "train")
        val_ds = load_split(self.data_root, "dev")
        raw = DatasetDict(train=train_ds, validation=val_ds)

        # Funtion to tokenize audio and encode text labels
        def prepare_batch(batch):
            
            #batch["path"]["array"] contains raw waveform
            audio = batch["path"]["array"]
            inputs = self.processor(audio, sampling_rate=16_000)
            
            # Feature extraction (waveform to input values)
            batch["input_values"] = inputs.input_values[0]
            batch["attention_mask"] = inputs.attention_mask[0]
            
            # Encode transcripts to label ids
            with self.processor.as_target_processor():
                batch["labels"] = self.processor(batch["sentence"]).input_ids
            return batch

        # Preprocess the dataset
        # Map the prepare_batch function to each example in the dataset
        self.dataset = raw.map(
            prepare_batch,
            remove_columns=["path", "sentence"],
            num_proc=os.cpu_count()
        )

    def train_dataloader(self):
        """
        Return a DataLoader for training.
        Uses dynamic padding collator.
        """
        collator = DataCollatorCTCWithPadding(
            processor=self.processor, 
            padding=True
        )
        return DataLoader(
            self.dataset["train"], 
            batch_size=self.batch_size,
            collate_fn=collator, 
            num_workers=self.num_workers, 
            shuffle=True
        )

    def val_dataloader(self):
        """
        Return a DataLoader for validation.
        """
        collator = DataCollatorCTCWithPadding(
            processor=self.processor,
            padding=True
        )
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.batch_size,
            collate_fn=collator, 
            num_workers=self.num_workers
        )

class ASRModule(L.LightningModule):
    """
    LightningModule wrapping Wav2Vec2ForCTC:
    - training_step logs CTC loss
    - validation_step computes WER and loss
    """
    def __init__(self, model_name: str, learning_rate: float = 3e-4,
                 warmup_steps: int = 500, total_steps: int = 10000):
        super().__init__()
        # Save hyperparameters for later use
        self.save_hyperparameters()
        
        # Load the Wav2Vec2 processor and model
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        
        # Freeze the feature extractor to prevent training on CNN layers
        self.model.freeze_feature_extractor()
        
        # WER metric for validation
        self.wer_metric = load_metric("wer")

    def training_step(self, batch, batch_idx):
        """Forward pass, compute CTC loss, log it."""
        
        outputs = self.model(input_values=batch["input_values"],
                             attention_mask=batch["attention_mask"],
                             labels=batch["labels"])
        loss = outputs.loss
        
        # Log the training loss per step and per epoch
        self.log("train_loss", 
                loss, 
                prog_bar=True, 
                on_step=True, 
                on_epoch=True
            )
        return loss

    def validation_step(self, batch, batch_idx):
        """Run forward, decode predictions, compute and log WER and loss."""
        
        outputs = self.model(input_values=batch["input_values"],
                             attention_mask=batch["attention_mask"],
                             labels=batch["labels"]
        )
        
        logits = outputs.logits
        pred_ids = torch.argmax(logits, dim=-1)
        
        # Decode predictions
        pred_str = self.processor.batch_decode(pred_ids)
        label_ids = batch["labels"].clone()
        
        # Prepare label strings (replace -100 with pad token id)
        # -100 is the label used for padding in the dataset
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(label_ids, group_tokens=False)
        
        # Compute WER
        wer = self.wer_metric.compute(predictions=pred_str, references=label_str)
        
        # Log WER and loss
        self.log("val_wer", wer, prog_bar=True)
        self.log("val_loss", outputs.loss, prog_bar=True)

    def configure_optimizers(self):
        """Set up AdamW optimizer and linear LR scheduler."""
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.total_steps
        )
        
        return [optimizer], [{
            "scheduler": scheduler, 
            "interval": "step"
        }]
        
# For reproducibility
L.seed_everything(42)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True,
                        help='Parent dir containing train/ and dev/')
    parser.add_argument('--model_name', type=str,
                        default='facebook/wav2vec2-large-xlsr-53')
    parser.add_argument('--output_dir', type=str, default='outputs/xlsr_from_dir')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    # Initialize the processor and data module
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    dm = ASRDataModule(data_root=args.data_root,
                       processor=processor,
                       batch_size=args.batch_size,
                       num_workers=args.num_workers)

    # Build the dataset
    dm.setup() 
    
    # Estimate total training steps for LR scheduler
    train_size = len(dm.dataset['train'])
    total_steps = (train_size // args.batch_size) * args.epochs

    # Initialize the LightningModule with hyprparameters
    model = ASRModule(model_name=args.model_name,
                      learning_rate=3e-4,
                      warmup_steps=500,
                      total_steps=total_steps)

    # Checkpoint callback monitors val_wer
    checkpoint_cb = ModelCheckpoint(monitor='val_wer', mode='min',
                                    filename='asr-{epoch:02d}-{val_wer:.3f}')

    # Configure the Trainer for multi-node/gpu DDP via SLURM
    trainer = L.Trainer(max_epochs=args.epochs,
                        accelerator='gpu',
                        devices=1,
                        num_nodes=int(os.environ.get('SLURM_NNODES', 1)),
                        strategy='ddp',
                        default_root_dir=args.output_dir,
                        precision='16-mixed',
                        callbacks=[checkpoint_cb])

    # Start training
    trainer.fit(model, dm)

    # Save final model
    trainer.save_checkpoint(os.path.join(args.output_dir, 'final.ckpt'))
    processor.save_pretrained(args.output_dir)