#!/usr/bin/env python
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from asr import (
    ASRModule,
    Wav2Vec2Processor,
    load_split,
    prepare_batch,
    DataCollatorCTCWithPadding,
)
from torchmetrics.text import WordErrorRate

if __name__ == "__main__":
    CHECKPOINT_PATH = "outputs/asr_run/final.ckpt"
    MODEL_NAME     = "/project/tb901149-tb0049/potae/Mamba/task1/wav2vec2-large-xlsr-53-chinese-zh-cn"
    DATA_ROOT      = "/project/tb901149-tb0049/asr_dataset"
    OUTPUT_CSV     = "submission.csv"
    BATCH_SIZE     = 8
    NUM_WORKERS    = 4

    # ─── 1) Load Lightning module, but ignore unexpected BnB keys ───────────
    asr: ASRModule = ASRModule.load_from_checkpoint(
        CHECKPOINT_PATH,
        model_name=MODEL_NAME,
        learning_rate=3e-4,
        warmup_steps=500,
        total_steps=10000,
        strict=False,    # ← allow extra BitsAndBytes keys to be skipped
    )
    asr.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr.to(device)

    # Grab the processor (tokenizer + feature_extractor)
    processor: Wav2Vec2Processor = asr.processor

    # ─── 2) Load & preprocess the “test” split exactly as in training ────────
    test_ds = load_split(DATA_ROOT, "test")
    test_ds = test_ds.map(
        lambda ex: prepare_batch(ex, processor),
        remove_columns=["path", "sentence"],
        num_proc=os.cpu_count(),
        load_from_cache_file=True,
    )

    dl = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,  # must be False so index stays in sync with raw_df
        num_workers=NUM_WORKERS,
        collate_fn=DataCollatorCTCWithPadding(processor),
    )

    # ─── 3) Read raw “test.csv” (contains at least “path”; maybe “sentence”) ───
    raw_df = pd.read_csv(os.path.join(DATA_ROOT, "test", "annotation", "test.csv"))

    # Prepare containers:
    records   = []
    all_preds = []
    all_refs  = []
    idx       = 0

    with torch.no_grad():
        for batch in dl:
            inp  = batch["input_values"].to(device)
            mask = batch["attention_mask"].to(device)

            out     = asr.model(input_values=inp, attention_mask=mask)
            logits  = out.logits                 # shape (B, T, V)
            pred_ids = torch.argmax(logits, dim=-1)  # shape (B, T)
            pred_strs = processor.batch_decode(pred_ids)

            B = pred_ids.size(0)
            for j in range(B):
                row       = raw_df.iloc[idx + j]
                audio_rel = row["path"]
                reference = row.get("sentence", "")  # if “sentence” exists in test.csv
                full_audio = os.path.join(DATA_ROOT, "test", "audio", audio_rel)
                pred      = pred_strs[j]

                # 4a) Collect for WER
                all_preds.append(pred)
                all_refs.append(reference)

                # 4b) Build one row of submission: exactly “path” + “sentence”
                records.append({
                    "path": full_audio,
                    "sentence": pred,
                })
            idx += B

    # ─── 5) Dump submission.csv with two columns: path,sentence ─────────────
    submission_df = pd.DataFrame(records)
    submission_df.to_csv(OUTPUT_CSV, index=False)
    print(f"➡︎ Saved {len(records)} lines to {OUTPUT_CSV}")

    # ─── 6) Compute WER: if raw_df has no “sentence”, load test_solution.csv ─
    if "sentence" not in raw_df.columns or raw_df["sentence"].isna().all():
        sol_df = pd.read_csv(os.path.join(DATA_ROOT, "test", "annotation", "test_solution.csv"))
        all_refs = sol_df["sentence"].tolist()

    wer = WordErrorRate()
    wer_score = wer(all_preds, all_refs).item()
    print(f"ℹ️  Final WER on test set: {wer_score:.4f}")

