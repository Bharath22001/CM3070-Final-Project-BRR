"""
Turns a reading-order sequence of Domain 1/2 outputs into the text that
Domain 3's language model sees.

Input is a list of positions, each either an int (a train_label I commit to)
or None (a sign is there but I can't name it). Nothing else, so this module
never depends on pixels or boxes.

Rendering:
    named sign        -> its OGSL sign name        e.g. LUGAL, |IGI.DIB|
    sign with no name -> MZL{number}               e.g. MZL851
    train_label 0     -> [...]
    None (unknown)    -> [x]
    restored sign     -> [NAME]                    e.g. [LUGAL]

I lead with the sign name, never readings[0], because OGSL lists readings
alphabetically. [...] means the clay itself is broken; [x] means my pipeline
couldn't read a sign it thinks is there, and I keep the two apart so the model
never mistakes my failures for ancient damage. Each position gets its own
marker, so "[x] [x] [x]" still says three signs are missing.

It returns the display string plus a parallel list of Slots, which the
faithfulness checks use to know exactly what the model was given.

Run the self-test from the project root:
    python -m cuneiform.domain3_generative.verbalise
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from cuneiform import config

# The markers. Change them here and nowhere else.
MASK_MARKER = "[x]"      # my pipeline couldn't read a sign it believes is there
BROKEN_MARKER = "[...]"  # the annotator recorded broken clay

# Slot kinds
KIND_SIGN = "sign"        # a named sign
KIND_UNNAMED = "unnamed"  # a real sign with no OGSL name, shown as MZL####
KIND_BROKEN = "broken"    # train_label 0
KIND_UNKNOWN = "unknown"  # None came in


def bare_name(text: str) -> str:
    """
    A displayed token with the restoration brackets removed. Both halves of the
    faithfulness reference set call this one function so they can't drift apart.
    """
    return text.replace("[", "").replace("]", "")


@dataclass(frozen=True)
class Slot:
    """
    One reading-order position, fully resolved. `text` is exactly the substring
    in the verbalised output, so slot i can be traced back without re-parsing.
    """
    index: int
    train_label: int | None
    text: str
    kind: str
    mzl: int | None = None
    unicode: str | None = None
    readings: tuple[str, ...] = ()
    status: str = "mask"
    # True only when readings[0] was checked to be the sign's own reading.
    canonical_reading_first: bool = False
    # True if the sign was inferred by the Domain 2 model.
    restored: bool = False

    @property
    def is_named(self) -> bool:
        """True only when a real OGSL sign name backs this position."""
        return self.kind == KIND_SIGN


@dataclass
class Verbalisation:
    """The verbalised sequence plus everything the evaluation will need."""
    text: str
    slots: list[Slot] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.slots)

    def counts(self) -> dict[str, int]:
        """How many positions of each kind."""
        out = {KIND_SIGN: 0, KIND_UNNAMED: 0, KIND_BROKEN: 0, KIND_UNKNOWN: 0}
        for s in self.slots:
            out[s.kind] += 1
        return out

    @property
    def names(self) -> set[str]:
        """
        The sign names legitimately present in this input, the reference set for
        the faithfulness check. It holds everything shown to the model,
        including MZL#### fallbacks, in both [NAME] and NAME form.
        """
        out = set()
        for s in self.slots:
            if s.kind in (KIND_SIGN, KIND_UNNAMED):
                out.add(s.text)
                if s.restored:
                    out.add(bare_name(s.text))
        return out

    @property
    def readable_fraction(self) -> float:
        """Share of positions with any identity at all (named or MZL)."""
        if not self.slots:
            return 0.0
        good = sum(1 for s in self.slots if s.kind in (KIND_SIGN, KIND_UNNAMED))
        return good / len(self.slots)

    @property
    def n_restored(self) -> int:
        """Positions whose identity came from Domain 2, not from the clay."""
        return sum(1 for s in self.slots if s.restored)


class SignLookup:
    """
    Reader for artifacts/domain3/sign_lookup.json. The file is a dict with
    "lookup", "alternates" and "notes", and lookup is keyed by train_label as a
    string, so the constructor checks that shape up front.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else config.D3_SIGN_LOOKUP
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Build it with "
                f"python -m cuneiform.domain3_generative.finalize_signlist"
            )
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "lookup" not in raw:
            raise ValueError(
                f"{self.path} is not the expected "
                f'{{"lookup": ..., "alternates": ...}} shape; '
                f"top-level keys were {list(raw)[:6]}"
            )
        self.lookup: dict[str, dict] = raw["lookup"]
        self.alternates: dict = raw.get("alternates", {})
        self.notes = raw.get("notes")

    def __len__(self) -> int:
        return len(self.lookup)

    def record(self, train_label: int) -> dict | None:
        return self.lookup.get(str(int(train_label)))


_DEFAULT_LOOKUP: SignLookup | None = None


def get_lookup() -> SignLookup:
    """Cached lookup, so a long run parses the JSON only once."""
    global _DEFAULT_LOOKUP
    if _DEFAULT_LOOKUP is None:
        _DEFAULT_LOOKUP = SignLookup()
    return _DEFAULT_LOOKUP


def resolve(train_label: int | None, index: int = 0,
            lookup: SignLookup | None = None, strict: bool = True,
            restored: bool = False) -> Slot:
    """
    One position to one Slot. strict=True raises on a train_label missing from
    the lookup, since that means something upstream is mis-wired; strict=False
    turns it into an unknown position instead.
    """
    if train_label is None:
        return Slot(index=index, train_label=None, text=MASK_MARKER,
                    kind=KIND_UNKNOWN, status="mask", restored=False)

    label = int(train_label)
    lk = lookup if lookup is not None else get_lookup()
    rec = lk.record(label)

    if rec is None:
        if strict:
            raise ValueError(
                f"train_label {label} is not in {lk.path.name}. The vocabulary "
                f"contract says this cannot happen -- check what produced this "
                f"sequence. Pass strict=False to treat it as unknown instead."
            )
        # Can't happen given the vocabulary, so I never mark this as restored.
        return Slot(index=index, train_label=label, text=MASK_MARKER,
                    kind=KIND_UNKNOWN, status="absent_from_lookup",
                    restored=False)

    status = rec.get("status", "")
    mzl = rec.get("mzl")
    ucun = rec.get("ucun")
    readings = tuple(rec.get("readings") or ())
    name = rec.get("sign")

    # train_label 0 is the annotators' "broken" placeholder, not a sign.
    if label == config.BROKEN_LABEL or status == "damage_placeholder":
        # Never restored: Domain 2 can't predict <brk>.
        return Slot(index=index, train_label=label, text=BROKEN_MARKER,
                    kind=KIND_BROKEN, mzl=mzl,
                    status=status or "damage_placeholder", restored=False)

    # A real sign OGSL can't name (7 signs, 0.71% of tokens): fall back to its MZL number.
    if not name:
        text = f"MZL{mzl}" if mzl else f"LABEL{label}"
        if restored:
            text = f"[{text}]"
        return Slot(index=index, train_label=label, text=text,
                    kind=KIND_UNNAMED, mzl=mzl, unicode=ucun,
                    readings=readings, status=status, restored=restored)

    text = name
    if restored:
        text = f"[{text}]"
    return Slot(index=index, train_label=label, text=text, kind=KIND_SIGN,
                mzl=mzl, unicode=ucun, readings=readings, status=status,
                canonical_reading_first=bool(rec.get("canonical_reading_first")),
                restored=restored)


def verbalise(labels: Sequence[int | None] | Iterable[int | None],
              lookup: SignLookup | None = None,
              strict: bool = True,
              restored: Sequence[bool] | None = None) -> Verbalisation:
    """
    The public entry point.

        >>> verbalise([119, None, 0, 68]).text
        'LUGAL [x] [...] MZL851'

    Accepts None or the string "<mask>" for an unknown position.
    """
    slots: list[Slot] = []
    labels_list = list(labels)
    if restored is not None:
        restored_list = list(restored)
        if len(restored_list) != len(labels_list):
            raise ValueError("restored mask must match labels length")
    else:
        restored_list = [False] * len(labels_list)
        
    for i, (raw, is_restored) in enumerate(zip(labels_list, restored_list)):
        if isinstance(raw, str):
            if raw.strip() in ("<mask>", MASK_MARKER, ""):
                raw = None
            else:
                raw = int(raw)
        slots.append(resolve(raw, index=i, lookup=lookup, strict=strict, restored=is_restored))
    return Verbalisation(text=" ".join(s.text for s in slots), slots=slots)


# Self-test: three gates, each of which can fail loudly

# A known worked example. If it stops reproducing, the lookup or the rules changed.
_SAMPLE_LABELS = [121, 48, 90, 161, 11, 107, 81, 13, 11, 15, 27, 39]
_SAMPLE_EXPECTED = "A\u2082 TU HU |IGI.DIB| A SI ME MA A LU\u2082 \u0160A\u2083 UD"


def _self_test() -> None:
    import sys
    try:
        # The default Windows console can't print subscripts.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 70)
    print("VERBALISE SELF-TEST")
    print("=" * 70)

    lk = get_lookup()
    print(f"  lookup   : {lk.path}")
    print(f"  records  : {len(lk)}")
    print(f"  alternates: {len(lk.alternates)}")

    by_status: dict[str, int] = {}
    for rec in lk.lookup.values():
        by_status[rec.get("status", "?")] = by_status.get(rec.get("status", "?"), 0) + 1
    print("  status census:")
    for k, v in sorted(by_status.items(), key=lambda kv: -kv[1]):
        print(f"      {k:<26} {v}")

    # Gate 1: the sample reproduces exactly
    print("\n" + "-" * 70)
    print("GATE 1  documented sample reproduces")
    print("-" * 70)
    got = verbalise(_SAMPLE_LABELS).text
    print(f"  labels   : {_SAMPLE_LABELS}")
    print(f"  expected : {_SAMPLE_EXPECTED}")
    print(f"  got      : {got}")
    gate1 = got == _SAMPLE_EXPECTED
    print(f"  --> {'PASS' if gate1 else 'FAIL'}")
    if not gate1:
        print("  !! IF THIS FAILED, IGNORE EVERY NUMBER BELOW. The lookup or")
        print("     the rendering rules have changed and nothing downstream")
        print("     of this file can be trusted.")

    # The four awkward cases
    print("\n" + "-" * 70)
    print("EDGE CASES (Including Restored marker)")
    print("-" * 70)
    edge = verbalise([119, 68, 374, 0, None], restored=[False, True, False, False, True])
    for s in edge.slots:
        print(f"  label {str(s.train_label):>5} -> {s.text:<10} "
              f"kind={s.kind:<8} status={s.status} restored={s.restored}")
    print(f"  joined   : {edge.text}")
    print(f"  counts   : {edge.counts()}")
    print(f"  names    : {sorted(edge.names)}")
    # Both forms of every restored sign must be in the reference set.
    print(f"  bare forms: "
          f"{sorted(bare_name(s.text) for s in edge.slots if s.restored)}")

    # Gate 2: every corpus label resolves
    print("\n" + "-" * 70)
    print("GATE 2  every train_label in the corpus resolves")
    print("-" * 70)
    corpus = config.D2_CORPUS_JSONL
    if not corpus.exists():
        print(f"  SKIPPED -- {corpus} not found.")
        return

    seen: dict[int, int] = {}
    n_lines = 0
    with corpus.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            n_lines += 1
            for t in json.loads(raw)["tokens"]:
                seen[t] = seen.get(t, 0) + 1

    missing = sorted(t for t in seen if lk.record(t) is None)
    total_tokens = sum(seen.values())
    print(f"  corpus lines   : {n_lines:,}")
    print(f"  sign tokens    : {total_tokens:,}")
    print(f"  distinct labels: {len(seen)}")
    print(f"  unresolvable   : {len(missing)}  {missing[:10]}")
    gate2 = not missing
    print(f"  --> {'PASS' if gate2 else 'FAIL'}")

    # Gate 3: token-weighted named coverage is about 99.29%
    print("\n" + "-" * 70)
    print("GATE 3  token-weighted coverage matches the recorded figure")
    print("-" * 70)
    named = unnamed = broken = 0
    for label, n in seen.items():
        slot = resolve(label, strict=False, lookup=lk)
        if slot.kind == KIND_SIGN:
            named += n
        elif slot.kind == KIND_UNNAMED:
            unnamed += n
        else:
            broken += n
    real = named + unnamed
    pct = 100.0 * named / real if real else 0.0
    print(f"  real sign tokens : {real:,}")
    print(f"  named            : {named:,}  ({pct:.2f}%)")
    print(f"  MZL fallback     : {unnamed:,}  ({100 - pct:.2f}%)")
    print(f"  broken placeholder: {broken:,} "
          f"({100.0 * broken / total_tokens:.1f}% of all positions)")
    gate3 = 99.0 <= pct <= 99.6
    print(f"  expected ~99.29% --> {'PASS' if gate3 else 'FAIL'}")

    # A real line, end to end
    print("\n" + "-" * 70)
    print("A REAL TEST-SPLIT LINE, VERBALISED")
    print("-" * 70)
    with corpus.open("r", encoding="utf-8") as fh:
        for raw in fh:
            rec = json.loads(raw)
            if rec.get("split") == "test" and 8 <= len(rec["tokens"]) <= 18:
                v = verbalise(rec["tokens"])
                print(f"  {rec['tablet']} {rec['view']} "
                      f"seg{rec['segm_idx']} line{rec['line_idx']}")
                print(f"  tokens : {rec['tokens']}")
                print(f"  text   : {v.text}")
                print(f"  counts : {v.counts()}")
                print(f"  readable: {v.readable_fraction:.0%}")
                break

    print("\n" + "=" * 70)
    print(f"GATE 1 {'PASS' if gate1 else 'FAIL'} | "
          f"GATE 2 {'PASS' if gate2 else 'FAIL'} | "
          f"GATE 3 {'PASS' if gate3 else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    _self_test()