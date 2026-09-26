"""
Builds training examples for my Domain 2 masked language model.

Three choices differ from standard RoBERTa training:
1. Span masking. Real damage comes in runs (median 4 signs), so I sample span
   lengths from the measured damage distribution and place each span whole.
2. Line blocks. Lines average only 8.8 signs, so I pack consecutive lines from
   the same tablet into one sequence for more context.
3. A 25% mask rate, closer to how much of a tablet ends up unnamed.

I always use <mask> (no 80/10/10 rule), because <mask> really does appear at
inference. With deterministic=True, masking is seeded per example so the
validation loss doesn't jump around between epochs.
"""

from __future__ import annotations

import json
import random

import torch
from torch.utils.data import Dataset

from cuneiform import config
from cuneiform.domain2_sequence.tokenizer import SignTokenizer

MAX_LEN = 128          # tokens per example, including <s> and </s>
MASK_RATE = 0.25       # target share of predictable positions to mask
MAX_SPAN = 27          # longest observed real damage run


def load_corpus(split: str) -> list[dict]:
    if not config.D2_CORPUS_JSONL.exists():
        raise FileNotFoundError(
            f"{config.D2_CORPUS_JSONL} not found. Run build_corpus.py first."
        )
    lines = []
    with open(config.D2_CORPUS_JSONL, "r", encoding="utf-8") as f:
        for raw in f:
            rec = json.loads(raw)
            if rec["split"] == split:
                lines.append(rec)
    return lines


def measure_span_distribution(records: list[dict]) -> dict[int, float]:
    """Measure the real damage run-length distribution from <brk> runs, as {length: probability}."""
    counts: dict[int, int] = {}
    for rec in records:
        run = 0
        for t in rec["tokens"]:
            if t == config.BROKEN_LABEL:
                run += 1
            elif run:
                counts[run] = counts.get(run, 0) + 1
                run = 0
        if run:
            counts[run] = counts.get(run, 0) + 1

    counts = {k: v for k, v in counts.items() if k <= MAX_SPAN}
    total = sum(counts.values())
    if total == 0:
        return {1: 1.0}
    return {k: v / total for k, v in sorted(counts.items())}


def build_blocks(records: list[dict], tokenizer: SignTokenizer,
                 max_len: int = MAX_LEN) -> list[list[int]]:
    """
    Pack consecutive lines from the same tablet into fixed-length sequences, in
    line order. A block never spans two tablets.
    """
    by_segment: dict[tuple, list[dict]] = {}
    for rec in records:
        key = (rec["tablet"], rec["view"], rec["segm_idx"])
        by_segment.setdefault(key, []).append(rec)

    budget = max_len - 2      # leave room for <s> and </s>
    blocks: list[list[int]] = []

    for key, recs in by_segment.items():
        recs.sort(key=lambda r: r["line_idx"])
        current: list[int] = []
        for rec in recs:
            ids = tokenizer.encode_line(rec["tokens"])
            if not ids:
                continue
            if len(ids) > budget:
                for i in range(0, len(ids), budget):
                    piece = ids[i:i + budget]
                    if len(piece) >= 4:
                        blocks.append(piece)
                continue
            if len(current) + len(ids) > budget:
                if len(current) >= 4:
                    blocks.append(current)
                current = list(ids)
            else:
                current.extend(ids)
        if len(current) >= 4:
            blocks.append(current)

    return blocks


class CuneiformMLMDataset(Dataset):
    """
    Yields (input_ids, attention_mask, labels) for masked language modelling.
    labels is -100 except at the positions I masked, so the loss only covers those.
    """

    def __init__(self, split: str, tokenizer: SignTokenizer,
                 span_dist: dict[int, float] | None = None,
                 max_len: int = MAX_LEN, mask_rate: float = MASK_RATE,
                 seed: int = 1729, deterministic: bool = False):
        self.tok = tokenizer
        self.max_len = max_len
        self.mask_rate = mask_rate
        self.split = split
        self.seed = seed
        self.deterministic = deterministic

        records = load_corpus(split)
        if not records:
            raise ValueError(f"No records found for split '{split}'.")

        self.span_dist = span_dist or measure_span_distribution(records)
        self.span_lengths = list(self.span_dist.keys())
        self.span_weights = list(self.span_dist.values())

        self.blocks = build_blocks(records, tokenizer, max_len)
        self.rng = random.Random(seed)

        n_tokens = sum(len(b) for b in self.blocks)
        n_brk = sum(sum(1 for t in b if t == tokenizer.brk_id) for b in self.blocks)
        n_unk = sum(sum(1 for t in b if t == tokenizer.unk_id) for b in self.blocks)
        flag = " [deterministic]" if deterministic else ""
        print(f"  [{split}]{flag} {len(self.blocks):,} blocks | {n_tokens:,} tokens | "
              f"{100 * n_brk / max(1, n_tokens):.1f}% <brk> | "
              f"{100 * n_unk / max(1, n_tokens):.2f}% <unk>")

    def __len__(self) -> int:
        return len(self.blocks)

    def _sample_span_length(self, rng: random.Random) -> int:
        return rng.choices(self.span_lengths, weights=self.span_weights, k=1)[0]

    def _candidate_starts(self, span_len: int, predictable_set: set[int],
                          masked: set[int], n: int) -> list[int]:
        """
        Find every start where a span of exactly `span_len` fits intact: all
        positions predictable and unmasked, with unmasked neighbours on both
        sides so two spans never merge into a longer one.
        """
        candidates = []
        for start in range(1, n - span_len):
            if (start - 1) in masked or (start + span_len) in masked:
                continue
            ok = True
            for j in range(start, start + span_len):
                if j not in predictable_set or j in masked:
                    ok = False
                    break
            if ok:
                candidates.append(start)
        return candidates

    def __getitem__(self, idx: int):
        # Seeded per index in deterministic mode, so val masking is the same every epoch.
        rng = random.Random(self.seed + idx) if self.deterministic else self.rng

        body = list(self.blocks[idx])
        ids = [self.tok.bos_id] + body + [self.tok.eos_id]
        n = len(ids)

        labels = [-100] * n

        predictable = [i for i, t in enumerate(ids) if self.tok.is_predictable(t)]
        predictable_set = set(predictable)
        target_count = int(len(predictable) * self.mask_rate)

        masked: set[int] = set()
        attempts = 0

        # Place spans whole or not at all, so their lengths stay exact. I accept
        # slightly fewer masks than the target in return.
        while len(masked) < target_count and attempts < 60:
            attempts += 1
            span_len = self._sample_span_length(rng)

            if len(masked) + span_len > target_count + 2:
                continue

            candidates = self._candidate_starts(span_len, predictable_set, masked, n)
            if not candidates:
                continue

            start = rng.choice(candidates)
            for j in range(start, start + span_len):
                masked.add(j)

        for j in masked:
            labels[j] = ids[j]          # remember the true sign
            ids[j] = self.tok.mask_id   # always <mask>

        attention = [1] * n
        pad_n = self.max_len - n
        if pad_n > 0:
            ids.extend([self.tok.pad_id] * pad_n)
            labels.extend([-100] * pad_n)
            attention.extend([0] * pad_n)

        return (
            torch.tensor(ids[:self.max_len], dtype=torch.long),
            torch.tensor(attention[:self.max_len], dtype=torch.long),
            torch.tensor(labels[:self.max_len], dtype=torch.long),
        )