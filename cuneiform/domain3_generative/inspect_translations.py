"""
Looks for English translations I could use as BLEU/ROUGE ground truth.

It checks two candidates: Oracc's index-tra.json (which turns out to be a
stemmed search index, not running text) and the oracc/catf repository's
`#tr.en:` lines.

Run from D:\\FYP:
    python -m cuneiform.domain3_generative.inspect_translations
"""

import json
import re
import ssl
import sys
import urllib.request
import zipfile
from collections import Counter

try:
    import truststore
except ImportError:
    truststore = None

from cuneiform import config

ZIP_PATH = config.D3_ORACC_DIR / "saao-saa01.zip"
CATF_URL = ("https://raw.githubusercontent.com/oracc/catf/master/"
            "saao-saa01.catf")
CATF_DEST = config.D3_ORACC_DIR / "saao-saa01.catf"


def _fix_console_encoding() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def http_get(url, timeout=120):
    ctx = (truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
           if truststore else None)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return resp.read()


# Candidate 1: the translation index
def inspect_index_tra():
    print("-" * 68)
    print("1. IS index-tra.json USABLE AS REFERENCE TRANSLATION TEXT?")
    print("-" * 68)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        raw = zf.read("saao/saa01/index-tra.json").decode("utf-8", "replace")
    doc = json.loads(raw)
    print(f"  top-level keys : {list(doc.keys())[:12]}")

    def describe(obj, path="", depth=0):
        if depth > 3:
            return
        if isinstance(obj, dict):
            keys = list(obj.keys())
            print(f"    {path or '<root>'}: dict, {len(keys)} keys, "
                  f"sample={keys[:6]}")
            for k in keys[:2]:
                describe(obj[k], f"{path}.{k}", depth + 1)
        elif isinstance(obj, list):
            print(f"    {path}: list, len={len(obj)}, "
                  f"first={str(obj[0])[:120] if obj else '-'}")
            if obj:
                describe(obj[0], f"{path}[0]", depth + 1)
        else:
            print(f"    {path}: {type(obj).__name__} = {str(obj)[:120]}")

    describe(doc)

    # Are the entries single stemmed words, or running text?
    words = re.findall(r'"([A-Za-z][A-Za-z\' -]{2,60})"', raw[:2_000_000])
    multi = [w for w in words if " " in w.strip()]
    print(f"\n  quoted strings sampled     : {len(words):,}")
    print(f"  containing a space (phrase): {len(multi):,} "
          f"({len(multi) / max(len(words), 1):.1%})")
    print(f"  sample single tokens : {[w for w in words if ' ' not in w][:20]}")
    print(f"  sample phrases       : {multi[:10]}")
    print("\n  VERDICT: running sentences -> usable. "
          "Isolated stemmed tokens -> NOT usable for BLEU.")


# Candidate 2: canonical ATF
def inspect_catf():
    print()
    print("-" * 68)
    print("2. DOES THE CANONICAL ATF CARRY #tr.en: TRANSLATIONS?")
    print("-" * 68)
    if not CATF_DEST.exists():
        print(f"  Downloading {CATF_URL}")
        try:
            CATF_DEST.write_bytes(http_get(CATF_URL))
        except Exception as exc:
            print(f"  FAILED: {exc}")
            return
    print(f"  {CATF_DEST}  ({CATF_DEST.stat().st_size / 1e6:.2f} MB)")

    text = CATF_DEST.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    print(f"  lines: {len(lines):,}")

    prefixes = Counter()
    for ln in lines:
        s = ln.strip()
        if s.startswith("#"):
            prefixes[s.split(":")[0][:18]] += 1
        elif s.startswith("@"):
            prefixes[s.split()[0]] += 1
    print("\n  Line-prefix census (top 25):")
    for k, v in prefixes.most_common(25):
        print(f"    {k:<20} {v:>7,}")

    tr = [(i, ln) for i, ln in enumerate(lines) if "#tr.en" in ln]
    print(f"\n  '#tr.en' lines: {len(tr):,}")
    for i, ln in tr[:12]:
        print(f"    {i + 1:>7} | {ln.strip()[:150]}")

    texts = [ln for ln in lines if ln.strip().startswith("&P")]
    print(f"\n  '&P' text headers: {len(texts):,}")
    for ln in texts[:6]:
        print(f"    {ln.strip()[:100]}")

    # One full text, to see how translation lines interleave
    start = next((i for i, ln in enumerate(lines)
                  if ln.strip().startswith("&P")), None)
    if start is not None:
        print("\n  --- first 45 lines of the first text ---")
        for ln in lines[start:start + 45]:
            print(f"    {ln[:150]}")


def check_catf_join():
    print()
    print("-" * 68)
    print("3. CATF JOIN AGAINST OUR CORPUS")
    print("-" * 68)
    if not CATF_DEST.exists():
        print("  No .catf file; skipping.")
        return
    import pandas as pd

    text = CATF_DEST.read_text(encoding="utf-8", errors="replace")
    catf_ids = set(re.findall(r"^&(P\d{6})", text, re.M))
    print(f"  P-numbers in catf: {len(catf_ids):,}")

    frames = [pd.read_csv(p)
              for p in sorted(config.TRANSLITERATIONS_DIR.glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    ours = set(df["tablet_CDLI"].astype(str).unique())
    print(f"  our tablets      : {len(ours):,}")
    print(f"  overlap          : {len(ours & catf_ids):,}")
    print(f"  sample overlap   : {sorted(ours & catf_ids)[:10]}")


def main() -> None:
    _fix_console_encoding()
    if not ZIP_PATH.exists():
        raise SystemExit(f"Not found: {ZIP_PATH}")
    print("=" * 68)
    print("LOCATING ENGLISH TRANSLATIONS")
    print("=" * 68)
    inspect_index_tra()
    inspect_catf()
    check_catf_join()
    print()
    print("=" * 68)


if __name__ == "__main__":
    main()