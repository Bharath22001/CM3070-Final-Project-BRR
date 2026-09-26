"""
Combines the vision (Domain 1) and language (Domain 2) posteriors at each
reading-order position:

    score(sign) = log P_vision(sign | pixels) + lambda * log P_LM(sign | context)

lambda = 0 is pure vision and a large lambda is pure language model. A
confident detector dominates, an uncertain one leaves the LM to decide, and a
confident but wrong detection can be outvoted, which a hard threshold can
never do. Everything is in log space to avoid underflow.

The LM posterior is leave-one-out: I re-mask each scored position on its own,
otherwise RoBERTa just copies the input token back and could never disagree
with vision. Both posteriors are over the 282 sign tokens only.

Run the self-test from the project root:
    python -m cuneiform.orchestrator.fusion
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import RobertaForMaskedLM

from cuneiform import config
from cuneiform.domain2_sequence.tokenizer import SignTokenizer

# Floor before taking a log, so log() never returns -inf.
LOG_FLOOR = 1e-12

CKPT_DIR = config.D2_DIR / "checkpoints" / "domain2_distilroberta_pretrained"


# What Domain 1 hands over

@dataclass
class Detection:
    """
    One reading-order position as the vision stage reports it.

    kind is "detected" (a usable class prediction), "unknown" (a box outside
    the 82 emit classes) or "missed" (no box, but a sign should be there).
    scores maps train_label to probability and can hold several candidates;
    if it is empty for a detection, {train_label: confidence} is used.
    """
    kind: str
    train_label: int | None = None
    confidence: float = 0.0
    scores: dict[int, float] = field(default_factory=dict)

    @property
    def is_informative(self) -> bool:
        """False when vision adds nothing and the LM decides alone."""
        return self.kind == "detected"


# Vocabulary support

def support_ids(tok: SignTokenizer) -> torch.Tensor:
    """Token ids that can be predicted: the 282 sign tokens."""
    return torch.tensor(sorted(tok.id_to_sign.keys()), dtype=torch.long)


# Vision posterior

def vision_posterior(detections: list[Detection], tok: SignTokenizer,
                     floor: float = 1e-6) -> torch.Tensor:
    """
    Turn detector output into a distribution over the vocabulary, one row per
    position, with mass only on sign tokens.

    A detection of class k with confidence c puts c on k and spreads 1 - c over
    the rest, so a confident detection is never infinitely strong. Unknown and
    missed positions get a uniform row, which leaves the LM to decide.
    """
    V = tok.vocab_size
    sup = support_ids(tok)
    n_sup = len(sup)

    out = torch.zeros(len(detections), V, dtype=torch.float64)

    for i, det in enumerate(detections):
        row = torch.zeros(V, dtype=torch.float64)
        named = 0.0

        if det.is_informative:
            scores = dict(det.scores)
            if not scores and det.train_label is not None:
                scores = {int(det.train_label): float(det.confidence)}
            for label, p in scores.items():
                token_id = tok.sign_to_id.get(int(label))
                if token_id is None:
                    # Can't happen for a legal emit class; skipping keeps a
                    # mis-wired detector from corrupting the row.
                    continue
                row[token_id] += float(p)
                named += float(p)

        residual = max(0.0, 1.0 - named)
        row[sup] += residual / n_sup
        row[sup] += floor
        out[i] = row / row.sum()

    return out


# Language-model posterior

def lm_posterior(model, input_ids: torch.Tensor, positions: list[int],
                 tok: SignTokenizer, batch_size: int = 32,
                 device: str = "cpu") -> torch.Tensor:
    """
    Leave-one-out LM distributions for `positions` within one block. For each
    position I copy the block, mask just that position and read the softmax,
    batching the copies. Returns [len(positions), vocab_size] probabilities.
    """
    model.eval()
    attn = (input_ids != tok.pad_id).long()
    out = torch.zeros(len(positions), tok.vocab_size, dtype=torch.float64)

    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            chunk = positions[start:start + batch_size]

            batch = input_ids.unsqueeze(0).repeat(len(chunk), 1).clone()
            for r, p in enumerate(chunk):
                batch[r, p] = tok.mask_id

            logits = model(
                input_ids=batch.to(device),
                attention_mask=attn.unsqueeze(0).repeat(len(chunk), 1).to(device),
            ).logits

            for r, p in enumerate(chunk):
                probs = torch.softmax(logits[r, p].float(), dim=-1)
                out[start + r] = probs.detach().cpu().to(torch.float64)

    return out


# The fusion itself

def fuse(p_vision: torch.Tensor, p_lm: torch.Tensor, lam: float,
         tok: SignTokenizer) -> torch.Tensor:
    """
    log P_vision + lambda * log P_LM over the sign tokens. Returns unnormalised
    scores; non-sign columns are -inf so the argmax can't pick a special token.
    """
    if p_vision.shape != p_lm.shape:
        raise ValueError(
            f"shape mismatch: vision {tuple(p_vision.shape)} "
            f"vs lm {tuple(p_lm.shape)}"
        )

    sup = support_ids(tok)
    log_v = torch.log(p_vision.clamp_min(LOG_FLOOR))
    log_l = torch.log(p_lm.clamp_min(LOG_FLOOR))

    scores = torch.full_like(log_v, float("-inf"))
    scores[:, sup] = log_v[:, sup] + lam * log_l[:, sup]
    return scores


def argmax_labels(scores: torch.Tensor, tok: SignTokenizer) -> list[int]:
    """Score matrix to one train_label per row."""
    ids = scores.argmax(dim=-1).tolist()
    return [tok.id_to_sign[i] for i in ids]


def fused_predictions(model, input_ids: torch.Tensor,
                      detections: dict[int, Detection], lam: float,
                      tok: SignTokenizer, device: str = "cpu",
                      lm_cache: torch.Tensor | None = None):
    """
    One block in, predictions out, scoring only the positions in `detections`.
    Returns (positions, predicted_labels, p_lm).

    The LM posterior doesn't depend on lambda, so passing lm_cache lets a
    lambda sweep reuse it. At lambda = 0 an uninformative position gives an
    arbitrary argmax; use Detection.is_informative to decide what to emit.
    """
    positions = sorted(detections)
    dets = [detections[p] for p in positions]

    p_v = vision_posterior(dets, tok)
    p_l = lm_cache if lm_cache is not None else lm_posterior(
        model, input_ids, positions, tok, device=device
    )

    scores = fuse(p_v, p_l, lam, tok)
    return positions, argmax_labels(scores, tok), p_l


def load_sequence_model(ckpt_dir=CKPT_DIR, device: str = "cpu"):
    """Load the trained Domain 2 checkpoint for inference."""
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"{ckpt_dir} not found. Train Domain 2 first, or point CKPT_DIR "
            f"at the right run."
        )
    model = RobertaForMaskedLM.from_pretrained(ckpt_dir)
    model.to(device)
    model.eval()
    return model


# Self-test

def _self_test() -> None:
    """
    Three checks on one real block from the test split: A, lambda = 0 gives the
    vision argmax exactly; B, a large lambda gives the LM argmax; C, raising
    lambda lets the LM overturn a confident but wrong detection.
    """
    from cuneiform.domain2_sequence.dataset import build_blocks, load_corpus

    print("=" * 68)
    print("FUSION SELF-TEST")
    print("=" * 68)

    tok = SignTokenizer()
    print(f"  {tok}")
    print(f"  support: {len(support_ids(tok))} sign tokens\n")

    model = load_sequence_model()
    print(f"  loaded {CKPT_DIR.name}\n")

    records = load_corpus("test")
    blocks = build_blocks(records, tok)

    # Pick a block with enough real signs, several of them detectable.
    chosen = None
    for b in blocks:
        predictable = [i for i, t in enumerate(b) if tok.is_predictable(t)
                       and t != tok.unk_id]
        emit = [i for i in predictable
                if tok.id_to_sign[b[i]] in tok.vision_emit_classes]
        if len(predictable) >= 12 and len(emit) >= 5:
            chosen = (b, predictable, emit)
            break
    if chosen is None:
        print("  No suitable block found; widen the criteria.")
        return

    body, _, _ = chosen
    ids = torch.tensor([tok.bos_id] + body + [tok.eos_id], dtype=torch.long)

    # A perfect but limited detector: emit-class signs at 0.90, the rest missed.
    detections: dict[int, Detection] = {}
    for i, t in enumerate(ids.tolist()):
        if not tok.is_predictable(t) or t == tok.unk_id:
            continue
        label = tok.id_to_sign[t]
        if label in tok.vision_emit_classes:
            detections[i] = Detection("detected", train_label=label, confidence=0.90)
        else:
            detections[i] = Detection("missed")

    informative = [p for p in sorted(detections) if detections[p].is_informative]
    print(f"  block: {len(ids)} tokens | {len(detections)} scored positions "
          f"| {len(informative)} detected, {len(detections) - len(informative)} missed")

    # The block as the LM receives it, masked wherever vision failed.
    masked = ids.clone()
    for p, det in detections.items():
        if not det.is_informative:
            masked[p] = tok.mask_id

    truth = {p: tok.id_to_sign[ids[p].item()] for p in detections}

    print("\n  computing leave-one-out LM posteriors "
          f"({len(detections)} forward passes)...")
    positions, pred0, p_lm = fused_predictions(model, masked, detections, 0.0, tok)
    cache = p_lm

    # A
    vision_top = {p: detections[p].train_label for p in informative}
    got = dict(zip(positions, pred0))
    ok_a = all(got[p] == vision_top[p] for p in informative)
    print(f"\n  A. lambda=0 reproduces vision argmax : "
          f"{'PASS' if ok_a else 'FAIL'}")

    # B
    lm_only = argmax_labels(
        fuse(torch.ones_like(cache) / cache.shape[1], cache, 1.0, tok), tok
    )
    _, pred_big, _ = fused_predictions(model, masked, detections, 1e6, tok,
                                       lm_cache=cache)
    ok_b = pred_big == lm_only
    print(f"  B. large lambda reproduces LM argmax : "
          f"{'PASS' if ok_b else 'FAIL'}")

    lm_acc = sum(1 for p, g in zip(positions, lm_only) if g == truth[p])
    print(f"     (LM alone is right at {lm_acc}/{len(positions)} positions here)")

    # C
    target = informative[0]
    true_label = truth[target]
    wrong = next(l for l in sorted(tok.vision_emit_classes) if l != true_label)
    detections[target] = Detection("detected", train_label=wrong, confidence=0.55)

    print(f"\n  C. injecting a confident-but-wrong detection at position {target}")
    print(f"     truth = sign {true_label} | detector says sign {wrong} @ 0.55")
    for lam in (0.0, 0.5, 1.0, 2.0, 4.0):
        _, preds, _ = fused_predictions(model, masked, detections, lam, tok,
                                        lm_cache=cache)
        got = dict(zip(positions, preds))[target]
        verdict = "recovered" if got == true_label else (
            "still wrong" if got == wrong else "changed, still wrong")
        print(f"     lambda={lam:<5} -> sign {got:<4} ({verdict})")

    print("\n" + "=" * 68)
    print("Checks A and B are the ones that must PASS. C is an observation:")
    print("whether the LM overturns this particular sign depends on context,")
    print("and the lambda sweep in evaluate_fusion.py measures it properly.")
    print("=" * 68)


if __name__ == "__main__":
    _self_test()