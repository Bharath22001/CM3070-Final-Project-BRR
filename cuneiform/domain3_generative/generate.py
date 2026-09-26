"""
Domain 3, my generative stage. It sends a verbalised sign sequence to a local
LLaMA-3.1 8B through Ollama and gets back a structured reading.

A general 8B model can't reliably translate a sequence of Neo-Assyrian sign
names with many positions missing, and would invent fluent English if asked.
So my prompt asks for a tentative reading with a self-assessed confidence,
forbids invention, and makes the model list the signs it used. That lets me
measure schema validity, faithfulness to the supplied signs, and how the
output degrades with damage.

Catalogue metadata is optional: 7 of my 32 test tablets have none, and which
fields exist depends on the volume, so I only show the fields that are
present. With no metadata, the prompt says so and forbids guessing author,
date or place. num_ctx is set explicitly because Ollama's default of 2048 can
silently cut off the start of the prompt on large tablets.

Run from the project root:
    python -m cuneiform.domain3_generative.generate
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field

from cuneiform import config
from cuneiform.domain3_generative.verbalise import (
    KIND_BROKEN, KIND_SIGN, KIND_UNKNOWN, KIND_UNNAMED, MASK_MARKER,
    BROKEN_MARKER,
    Verbalisation, bare_name, verbalise,
)

MODEL = "llama3.1:8b"
NUM_CTX = 8192
TEMPERATURE = 0.2   # low, for a cautious reading rather than creative writing

# Cap on generated tokens. Without it two calls looped inside one JSON string
# for minutes. 4096 leaves room for long citation lists (one entry per
# occurrence) while still stopping a runaway.
NUM_PREDICT = 4096

# Metadata fields worth showing, in reading order. Only populated ones are emitted.
METADATA_FIELDS = [
    ("designation", "Publication reference"),
    ("genre", "Genre"),
    ("period", "Period"),
    ("dialect", "Dialect"),
    ("provenience", "Find site"),
    ("archive", "Archive"),
    ("ancient_author", "Ancient sender"),
    ("ancient_recipient", "Ancient recipient"),
    ("collection", "Held by"),
    ("museum_no", "Museum number"),
    ("primary_publication", "Primary publication"),
]

# The output schema. Ollama enforces it through its `format` parameter, so this
# stage returns machine-readable JSON.
SCHEMA = {
    "type": "object",
    "properties": {
        "tentative_reading": {"type": "string"},
        "confidence": {"type": "string",
                       "enum": ["none", "low", "medium", "high"]},
        "signs_used": {"type": "array", "items": {"type": "string"}},
        "uncertainty_notes": {"type": "string"},
        "historical_context": {"type": "string"},
        "metadata_used": {"type": "boolean"},
    },
    "required": ["tentative_reading", "confidence", "signs_used",
                 "uncertainty_notes", "historical_context", "metadata_used"],
}

SYSTEM_PROMPT = """\
You are assisting an Assyriologist reading a Mesopotamian clay tablet.

You are given a transliteration expressed as CUNEIFORM SIGN NAMES in reading
order, one physical line of the tablet per line of input. Some signs carry a
conventional reading in parentheses, e.g. LUGAL (lugal).

Three markers appear in the input and they mean DIFFERENT things:
  [...]  the original annotator recorded that the clay here is broken. This is
         evidence about the tablet itself.
  [x]    the automated reading pipeline believes a sign is present here but
         could not identify it. This is a limitation of the system, not damage.
  [NAME] square brackets around a sign name mean it was restored by a statistical 
         model, not read from the tablet. Treat it as less certain.
Each marker stands for exactly ONE sign. Never merge a run of markers.

YOUR TASK IS TO READ WHAT IS THERE.
Damaged tablets are the normal case, not an obstacle to refuse. If most
positions carry a sign name, you are expected to work with them: identify
recurring formulae, numerals, measures, personal-name markers, and the apparent
structure and purpose of the text, and say what the legible signs support.
Refusing to engage with a largely legible tablet is a FAILURE, not caution.

CALIBRATE CONFIDENCE TO LEGIBILITY:
  high    - not appropriate here; you are reading sign names, not a known text
  medium  - most positions legible AND recognisable structure or formulae
  low     - partial structure only, or heavy damage
  none    - almost nothing legible
Reserve "this cannot be read" for input that is genuinely mostly markers.

RULES YOU MUST FOLLOW:
1. Do not invent content. Never supply a translation more specific than the
   signs support, and never introduce signs, words, kings, places or proper
   names that are not derivable from the input.
2. "signs_used" must list the DISTINCT input sign names you actually relied on.
   List each sign name AT MOST ONCE, however many times it appears on the
   tablet, and copy it exactly as written in the input. This is the evidence
   for your reading, not an inventory of the tablet. If your confidence is
   anything other than "none", this list must NOT be empty.
3. Distinguish what the signs SHOW from what you INFER. Put inferences, and
   what would resolve them, in "uncertainty_notes".
4. Base "historical_context" ONLY on catalogue metadata supplied below. If no
   metadata is supplied, say that none was available and do NOT guess at the
   date, author, archive, provenance or purpose of the tablet.

Reply with JSON conforming to the requested schema and nothing else.\
"""


# Catalogue metadata

def load_catalogue() -> dict[str, dict]:
    """Merge every extracted Oracc catalogue on disk, keyed by P-number."""
    merged: dict[str, dict] = {}
    for d in sorted(config.D3_ORACC_DIR.glob("saao-*")):
        f = d / "catalogue.json"
        if d.is_dir() and f.exists():
            doc = json.loads(f.read_text(encoding="utf-8"))
            for k in ("members", "entries", "texts"):
                if isinstance(doc.get(k), dict):
                    for tid, rec in doc[k].items():
                        merged.setdefault(tid, rec)
                    break
    return merged


def metadata_block(tablet: str, catalogue: dict[str, dict]) -> tuple[str, bool]:
    """Render the catalogue block for one tablet, populated fields only. Returns (text, had_metadata)."""
    rec = catalogue.get(tablet)
    lines = []
    if rec:
        for key, label in METADATA_FIELDS:
            val = rec.get(key)
            if val not in (None, "", []):
                lines.append(f"  {label}: {val}")
    if not lines:
        return ("CATALOGUE METADATA: none available for this tablet.\n"
                "Do not speculate about its date, author, archive or "
                "provenance.", False)
    return "CATALOGUE METADATA (CC0, Oracc SAAo):\n" + "\n".join(lines), True


# Prompt assembly

def display_tokens(v: Verbalisation) -> list[str]:
    """
    One token per position, exactly as it appears in the prompt. A reading is
    only shown when readings[0] is known to be the sign's own reading. Kept
    separate from gloss() because the faithfulness check needs exactly what the
    model saw.
    """
    out = []
    for s in v.slots:
        if s.kind == KIND_SIGN and s.canonical_reading_first and s.readings:
            out.append(f"{s.text} ({s.readings[0]})")
        else:
            out.append(s.text)
    return out


def gloss(v: Verbalisation) -> str:
    """One tablet line to its display string."""
    return " ".join(display_tokens(v))


def normalise_citation(s: str) -> str:
    """
    Strip notation from a sign name, leaving its identity: the [ ] restoration
    brackets and the | | compound delimiters. Citing "ME.U.U.U" for an input
    showing "|ME.U.U.U|" is a notation choice, not an invention, while a real
    invention like "U.U" still fails to match.
    """
    return bare_name(s).replace("|", "").strip()


def classify_citations(cited, supplied: set[str]):
    """
    Split citations into exact, normalised and invented. I keep the middle tier
    separate so it shows how often the model renames what it saw. `normalised`
    holds (citation, matched form) pairs.
    """
    by_norm: dict[str, str] = {}
    for s in supplied:
        by_norm.setdefault(normalise_citation(s), s)

    exact, normalised, invented = [], [], []
    for c in cited:
        if c in supplied:
            exact.append(c)
        elif normalise_citation(c) in by_norm:
            normalised.append((c, by_norm[normalise_citation(c)]))
        else:
            invented.append(c)
    return exact, normalised, invented


# The prose scan. One shared function, used by the CLI, the API and the evaluation.

PROSE_SCAN_FIELDS = ("tentative_reading", "uncertainty_notes")

# A |COMPOUND| sign, or a bare sign-shaped token. The character class covers
# what OGSL names contain: Š, subscript digits, ASCII digits, ×, @ and dots.
# evaluate_domain3 imports this same pattern.
PROSE_SIGN_RE = (
    r"\|[^|]+\|"
    r"|(?<![0-9A-Za-z\u0160])[A-Z\u0160][A-Z\u0160\u2080-\u20890-9\u00d7@.]+"
)

# Capitalised words that aren't signs. NAME is my own [NAME] restoration marker,
# the rest are abbreviations and initialisms. Kept short on purpose, since AD,
# BA and DI are real sign names.
PROSE_STOP_WORDS = {
    "JSON", "SAA", "ADD", "BM", "UK", "US", "CC0", "MZL", "NAME",
    "II", "III", "IV", "VI", "VII", "VIII", "IX", "XI", "XII",
    "U.S", "U.K", "U.S.A", "A.D", "B.C", "B.C.E", "C.E", "N.B",
}


def scan_prose(out: dict, supplied: set[str]) -> dict:
    """
    Find sign-shaped tokens in the prose fields and report which were never in
    the input. This catches signs invented inside the reading itself, which the
    citation list check can't see. It only sees sign-shaped tokens, so an
    invented king or city would pass. Returns {"fields", "candidates", "unmatched"}.
    """
    prose = " ".join(str(out.get(k, "")) for k in PROSE_SCAN_FIELDS)
    candidates = set(re.findall(PROSE_SIGN_RE, prose))
    norm_supplied = {normalise_citation(s) for s in supplied}
    # The pattern can grab a sentence's full stop; I strip it for the comparison only.
    unmatched = sorted(
        c for c in candidates
        if c.rstrip(".") not in supplied
        and normalise_citation(c.rstrip(".")) not in norm_supplied
        and c.rstrip(".") not in PROSE_STOP_WORDS
    )
    return {
        "fields": list(PROSE_SCAN_FIELDS),
        "candidates": sorted(candidates),
        "unmatched": unmatched,
    }


@dataclass
class TabletInput:
    """Everything Domain 3 needs about one tablet. No pixels, no boxes."""
    tablet: str
    lines: list[Verbalisation] = field(default_factory=list)

    @property
    def all_names(self) -> set[str]:
        """Sign names legitimately present, the faithfulness reference set."""
        names: set[str] = set()
        for v in self.lines:
            names |= v.names
        return names

    @property
    def displayed_names(self) -> set[str]:
        """
        Exactly the sign tokens the model sees, glosses included ("A (a)").
        Together with all_names this is the faithfulness reference set.
        """
        out: set[str] = set()
        for v in self.lines:
            for tok, s in zip(display_tokens(v), v.slots):
                if s.kind in (KIND_SIGN, KIND_UNNAMED):
                    out.add(tok)
                    if s.restored:
                        # Also add the bare form, since a model citing "LUGAL"
                        # for "[LUGAL] (lugal)" is being faithful.
                        out.add(bare_name(tok))
        return out

    @property
    def counts(self) -> dict[str, int]:
        total: dict[str, int] = {}
        for v in self.lines:
            for k, n in v.counts().items():
                total[k] = total.get(k, 0) + n
        return total

    @property
    def damage_rate(self) -> float:
        """
        Share of positions with no identity at all (broken plus unreadable).
        Restorations don't count here, so on a live tablet read restored_rate too.
        """
        c = self.counts
        n = sum(c.values())
        return (c.get(KIND_BROKEN, 0) + c.get(KIND_UNKNOWN, 0)) / n if n else 0.0

    @property
    def restored_rate(self) -> float:
        """Share of positions named by Domain 2 rather than read off the clay."""
        n = sum(self.counts.values())
        return self.n_restored / n if n else 0.0

    @property
    def n_restored(self) -> int:
        return sum(v.n_restored for v in self.lines)


def build_prompt(t: TabletInput, catalogue: dict[str, dict]) -> tuple[str, bool]:
    meta, had = metadata_block(t.tablet, catalogue)
    body = "\n".join(f"  {i + 1:>3}: {gloss(v)}" for i, v in enumerate(t.lines))
    c = t.counts
    n = sum(c.values())

    # A restored sign has a name but no evidence, so it doesn't count as legible.
    restored = t.n_restored
    read = n - c.get(KIND_BROKEN, 0) - c.get(KIND_UNKNOWN, 0) - restored
    pct = 100 * read / n if n else 0.0

    # Keep the ground-truth wording exactly as it was so earlier results stay
    # comparable; ground-truth tablets never have restorations.
    if restored:
        legibility = f"{read} ({pct:.0f}%) were read directly from the tablet."
        restored_line = (f"\n{restored} were restored by the sequence model "
                         f"[NAME] and are less certain.")
    else:
        legibility = f"{read} ({pct:.0f}%) carry a legible sign name."
        restored_line = ""

    prompt = f"""\
TABLET: {t.tablet}

{meta}

TRANSLITERATION: {len(t.lines)} lines, {n} sign positions, of which
{legibility}
{c.get(KIND_BROKEN, 0)} are broken [...] and {c.get(KIND_UNKNOWN, 0)} \
unidentified [x].{restored_line}

{body}

Produce the JSON object now."""
    return prompt, had


# The Ollama call

def generate(t: TabletInput, catalogue: dict[str, dict] | None = None,
             model: str = MODEL) -> dict:
    """
    One tablet in, one parsed JSON dict out. Raises on a malformed response
    instead of repairing it, since structural validity is something I measure.
    """
    import ollama

    catalogue = load_catalogue() if catalogue is None else catalogue
    prompt, had_meta = build_prompt(t, catalogue)

    resp = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],
        format=SCHEMA,
        options={"num_ctx": NUM_CTX, "temperature": TEMPERATURE,
                 "num_predict": NUM_PREDICT},
    )
    raw = resp["message"]["content"]

    # Ollama says why it stopped: "length" means the cap cut the JSON short,
    # "stop" means the model finished. The error says which.
    try:
        done = resp.get("done_reason")
    except AttributeError:
        done = getattr(resp, "done_reason", None)

    try:
        out = json.loads(raw)
    except json.JSONDecodeError as exc:
        why = ("TRUNCATED AT CAP -- raise NUM_PREDICT"
               if done == "length" else
               f"MALFORMED (done_reason={done!r})")
        raise ValueError(
            f"{why}: {exc} "
            f"[raw_len={len(raw):,} chars, num_predict={NUM_PREDICT}]"
        ) from exc

    if done == "length":
        # Parsed, but the model was still writing, so a field was cut short.
        out["_hit_cap"] = True
    out["_had_metadata"] = had_meta
    return out


# Loading a tablet from the corpus (ground-truth mode)

def load_tablet(tablet: str, mask_rate: float = 0.0,
                seed: int = 1729) -> TabletInput:
    """
    Build a TabletInput from the corpus. mask_rate > 0 replaces that share of
    real signs with unknown positions, to test Domain 3 on its own against
    upstream damage.
    """
    import random
    # Seeded with a string so the mask is the same in every process.
    rng = random.Random(f"{seed}|{tablet}|{mask_rate}")

    rows = []
    with config.D2_CORPUS_JSONL.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            if rec["tablet"] == tablet:
                rows.append(rec)
    if not rows:
        raise ValueError(f"tablet {tablet} not found in the corpus")

    rows.sort(key=lambda r: (r["view"], r["segm_idx"], r["line_idx"]))

    lines = []
    for rec in rows:
        toks: list[int | None] = []
        for tok in rec["tokens"]:
            if tok != config.BROKEN_LABEL and mask_rate > 0 \
                    and rng.random() < mask_rate:
                toks.append(None)
            else:
                toks.append(tok)
        lines.append(verbalise(toks))
    return TabletInput(tablet=tablet, lines=lines)


# Self-test

def _self_test() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 70)
    print("DOMAIN 3 GENERATE -- SELF-TEST")
    print("=" * 70)

    catalogue = load_catalogue()
    print(f"  catalogue records : {len(catalogue):,}")

    COVERED = "P335561"     # SAA 06, metadata present
    UNCOVERED = "P336128"   # CompVis annotated set, no catalogue entry

    # Prompt inspection, no LLM yet
    print("\n" + "-" * 70)
    print(f"1. PROMPT FOR {COVERED} (metadata expected)")
    print("-" * 70)
    t = load_tablet(COVERED)
    prompt, had = build_prompt(t, catalogue)
    print(f"  lines {len(t.lines)} | counts {t.counts} | "
          f"damage {t.damage_rate:.0%} | restored {t.n_restored} | "
          f"metadata {had}")
    print(f"  approx prompt chars: {len(prompt):,}")
    print("\n" + prompt[:1400])
    print("  ...")

    print("\n" + "-" * 70)
    print(f"2. METADATA BLOCK FOR {UNCOVERED} (none expected)")
    print("-" * 70)
    t2 = load_tablet(UNCOVERED)
    meta2, had2 = metadata_block(UNCOVERED, catalogue)
    print(f"  metadata present: {had2}")
    print("  " + meta2.replace("\n", "\n  "))

    # Gate: does the model return schema-valid JSON?
    print("\n" + "-" * 70)
    print("3. LIVE CALL -- first call after idle takes ~1 min (model load)")
    print("-" * 70)
    try:
        out = generate(t, catalogue)
    except Exception as exc:
        print(f"  CALL FAILED: {type(exc).__name__}: {exc}")
        print("  !! IF THIS FAILED, IGNORE EVERYTHING BELOW.")
        print("     Check the Ollama tray app is running.")
        return

    ok_schema = all(k in out for k in SCHEMA["required"])
    print(f"  schema-conformant : {'PASS' if ok_schema else 'FAIL'}")
    print(f"  confidence        : {out.get('confidence')}")
    print(f"  metadata_used     : {out.get('metadata_used')}")
    print(f"\n  tentative_reading :\n    {out.get('tentative_reading','')[:400]}")
    print(f"\n  uncertainty_notes :\n    {out.get('uncertainty_notes','')[:300]}")
    print(f"\n  historical_context:\n    {out.get('historical_context','')[:400]}")

    # Gate: faithfulness
    print("\n" + "-" * 70)
    print("4. FAITHFULNESS -- were all cited signs actually in the input?")
    print("-" * 70)
    supplied = t.all_names | t.displayed_names
    raw_cited = [s.strip() for s in out.get("signs_used", []) if s.strip()]
    cited = sorted(set(raw_cited))   # the model repeats a sign for each occurrence
    exact, normalised, invented = classify_citations(cited, supplied)
    print(f"  distinct signs supplied : {len(supplied)}")
    print(f"  citations returned      : {len(raw_cited)} "
          f"({len(cited)} distinct)")
    print(f"  cited exactly           : {len(exact)}")
    # Print what each normalised citation matched, not just the count.
    print(f"  cited after normalising : {len(normalised)}")
    for c, matched in normalised[:8]:
        print(f"      {c}  ->  {matched}")
    print(f"  NOT in input            : {len(invented)}  {invented[:12]}")
    if cited:
        print(f"  coverage of input       : "
              f"{len(set(cited) & supplied) / len(supplied):.0%}")
    # An empty citation list means the check never ran, which is not a pass.
    if not cited:
        print("  --> VACUOUS: nothing cited, so nothing could be checked.")
        print("      This is a gate failure, not a model success.")
    elif invented:
        print(f"  --> FAIL: {len(invented)} of {len(cited)} distinct citations "
              f"are not in the input.")
    else:
        print("  --> PASS: every cited sign was present in the input.")

    import re
    print("\n  PROSE SCAN (the text a researcher actually reads):")
    prose = " ".join(str(out.get(k, "")) for k in
                     ("tentative_reading", "uncertainty_notes"))
    # Sign names are upper-case runs, maybe subscripted, or |COMPOUND|.
    candidates = set(re.findall(r"\|[^|]+\||\b[A-Z][A-Z\u2080-\u2089]{1,}\b",
                                prose))
    # Strip wrappers before comparing and ignore ordinary English capitals.
    STOP = {"JSON", "SAA", "ADD", "BM", "UK", "CC0", "MZL"}
    unmatched = sorted(c for c in candidates
                       if c not in supplied and c.strip("|") not in supplied
                       and c not in STOP)
    print(f"    sign-shaped tokens in prose : {len(candidates)}")
    print(f"    NOT in the input            : {len(unmatched)}  {unmatched[:12]}")
    print("    (read the prose by eye as well -- a WRONG GLOSS of a REAL sign,")
    print("     e.g. calling URUDA 'dates', will not appear in this list.)")

    # Gate: restraint when there is no metadata
    print("\n" + "-" * 70)
    print(f"5. RESTRAINT -- {UNCOVERED} has no metadata. Does it speculate?")
    print("-" * 70)
    try:
        out2 = generate(t2, catalogue)
        ctx = out2.get("historical_context", "")
        # The model's own metadata_used claim isn't reliable, so the pipeline
        # uses my _had_metadata flag instead.
        claim = out2.get("metadata_used")
        truth = out2.get("_had_metadata")
        agree = (bool(claim) == bool(truth))
        print(f"  metadata_used claimed: {claim} | actual: {truth} | "
              f"{'agrees' if agree else 'CONTRADICTS'}")
        print(f"  confidence        : {out2.get('confidence')} "
              f"(tablet damage {t2.damage_rate:.0%})")
        print(f"  historical_context:\n    {ctx[:500]}")
        print("\n  READ THIS BY EYE: it should say no metadata was available.")
        print("  If it names an archive, a king, or a date, that is a")
        print("  hallucination and the prompt needs tightening.")
    except Exception as exc:
        print(f"  CALL FAILED: {exc}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    _self_test()