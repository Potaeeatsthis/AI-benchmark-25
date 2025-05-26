#!/usr/bin/env python3
"""
train_asr_from_dir.py

Fine-tune XLSR-53 on AI-Thailand Meeting Transcription data laid out as:

datasets/
  train/
    audio/…        ← your .mp3 files
    annotation/
      train.csv     ← columns: path,sentence
  dev/
    audio/…
    annotation/
      dev.csv

Usage:
  python train_asr_from_dir.py \
    --data_root ./datasets \
    --model_name facebook/wav2vec2-large-xlsr-53 \
    --output_dir outputs/xlsr_from_dir \
    --batch_size 8 \
    --epochs 5
"""

import argparse
import os
import pandas as pd
import torch

from datasets import Dataset, load_metric, Audio, DatasetDict
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    DataCollatorCTCWithPadding,
    Trainer,
    TrainingArguments,
)

def load_split(root: str, split: str) -> Dataset:
    """Load split CSV and attach full audio path."""
    
    csv_path = os.path.join(root, split, "annotation", f"{split}.csv")
    df = pd.read_csv(csv_path, names=["path","sentence"])
    
    # prefix the audio directory to each relative path
    audio_dir = os.path.join(root, split, "audio")
    df["path"] = df["path"].apply(lambda p: os.path.join(audio_dir, p))
    
    # build a dataset
    ds = Dataset.from_pandas(df, preserve_index=False)
    
    # cast the audio column to Audio
    ds = ds.cast_column("path", Audio(sampling_rate=16_000))
    
    return ds

def prepare_batch(batch, processor):
    """For each example, tokenize raw audio and encode transcript to labels."""
    
    audio = batch["path"]["array"]
    inputs = processor(audio, sampling_rate=16_000)
    batch["input_values"]   = inputs.input_values[0]
    batch["attention_mask"] = inputs.attention_mask[0]

    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    
    return batch

def compute_wer(pred, processor, wer_metric):
    """Compute WER metric for the predictions.
        wer = WER(predictions, references)
        where predictions = processor.batch_decode(pred.predictions.argmax(-1))
        and references = processor.batch_decode(label_ids, group_tokens=False)
        
    """
    
    pred_logits = pred.predictions.argmax(-1)
    pred_str = processor.batch_decode(pred_logits)
    # replace -100 in labels as pad_token_id
    label_ids = pred.label_ids.copy()
    # -100 is the label used for padding in the dataset
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)
    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                        help="Path to `datasets/` directory")
    parser.add_argument("--model_name", default="facebook/wav2vec2-large-xlsr-53")
    parser.add_argument("--output_dir", default="wav2vec2_xlsr_from_dir")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    # Load processor & model
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    
    # CTC (connected Time Classification) the way to train model for sequence-to-sequence tasks
    model     = Wav2Vec2ForCTC.from_pretrained(args.model_name)
    model.freeze_feature_extractor()

    # Load train + dev
    train_ds = load_split(args.data_root, "train")
    dev_ds   = load_split(args.data_root, "dev")
    dataset  = DatasetDict({"train": train_ds, "validation": dev_ds})

    # Preprocess: tokenize audio and encode labels
    dataset = dataset.map(
        lambda b: prepare_batch(b, processor),
        remove_columns=["path","sentence"],
        num_proc=os.cpu_count()
    )

    # Data collator for dynamic padding
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # WER metric
    wer_metric = load_metric("wer")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps", # evaluate every `eval_steps`
        eval_steps=500, #stop training every 500 steps to evaluate
        logging_steps=250, # log every 250 steps
        save_steps=500, # save model every 500 steps
        save_total_limit=2,
        learning_rate=3e-4,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="wer",
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset= dataset["validation"],
        data_collator= data_collator,
        tokenizer=      processor.feature_extractor,
        compute_metrics=lambda p: compute_wer(p, processor, wer_metric)
    )

    # Train & save
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Model and processor saved to {args.output_dir}")

if __name__ == "__main__":
    main()