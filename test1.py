import argparse
import os

import torch
from datasets import load_dataset, load_metric, Audio
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Trainer,
    TrainingArguments,
    DataCollatorCTCWithPadding,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_manifest", type=str, required=True,
                   help="CSV with columns [audio_path, transcript]")
    p.add_argument("--model_name_or_path", type=str, default="facebook/wav2vec2-base-960h")
    p.add_argument("--output_dir", type=str, default="wav2vec2_finetuned")
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=5)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()

    # Load manifest CSV into a Dataset
    dataset = load_dataset(
        "csv",
        data_files={"train": args.data_manifest},
        column_names=["audio_path", "transcript"],
    )

    # Cast audio column, resample to 16 kHz (common rate for speech recognition models like Wav2Vec2)
    dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16_000))
    dataset = dataset.rename_column("audio_path", "audio")

    # Load processor & model
    processor = Wav2Vec2Processor.from_pretrained(args.model_name_or_path)
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name_or_path,
        gradient_checkpointing=True,
    )
    model.freeze_feature_extractor() # to save memory, speed up training and over fitting
    
    # Preprocessing 
    def prepare_example(batch):
        audio = batch["audio"]["array"]
        
        # tokenize audio
        inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")
        batch["input_values"] = inputs.input_values[0]
        batch["attention_mask"] = inputs.attention_mask[0]

        # encode transcript
        with processor.as_target_processor():
            labels = processor(batch["transcript"]).input_ids
        batch["labels"] = labels
        return batch

    # Apply preprocessing (use multiple processes if large)
    dataset = dataset.map(
        prepare_example,
        remove_columns = ["audio", "transcript"],
        num_proc = os.cpu_count(), # or num_proc = 1, batch = Flase
        # batched = True,
        # batch_size = 16,
    )

    # Data collator for padding
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # WER (word error rate) metric
    wer_metric = load_metric("wer")
    
    '''
    
    Measures how many words in the predicted transcript are incorrect compared to the reference transcript.
    
    wer = (Substitutions + Insertions + Deletions) / Total words in reference
    
    '''

    def compute_metrics(pred):
        pred_logits = pred.predictions  # (batch, time, vocab)
        pred_ids = pred_logits.argmax(axis=-1)
        
        # decode predictions & labels
        preds = processor.bWERatch_decode(pred_ids)
        labels = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=preds, references=labels)
        return {"wer": wer}

    # Training arguments
    training_args = TrainingArguments(
        output_dir = args.output_dir, # model save path
        overwrite_output_dir = True,  # overwrite existing model
        per_device_train_batch_size = args.per_device_train_batch_size, # batch size for training
        per_device_eval_batch_size = args.per_device_eval_batch_size, # batch size for evaluation
        evaluation_strategy = "steps", # evaluate every N steps
        eval_steps = args.eval_steps, # how often to run evaluation
        logging_steps = args.logging_steps, # how often to log training info
        save_steps = args.eval_steps, # how often to save checkpoints
        learning_rate = args.learning_rate, # learning rate for optimizer
        num_train_epochs = args.num_train_epochs, # number of training epochs
        weight_decay = 0.005, # weight decay for regularization 
        warmup_steps = 500, # steps to gradually increase LR at start
        save_total_limit = 2, # max number of checkpoints to keep
        fp16 = torch.cuda.is_available(), # use mixed precision if GPU supports it
        gradient_checkpointing = True, # save memory by checkpointing
        load_best_model_at_end = True, # restore best model after training
        metric_for_best_model = "wer", # use WER to select best model
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["train"],  
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train!
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
