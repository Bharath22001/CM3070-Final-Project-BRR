"""
Downloads one Oracc SAAo volume (CC0) and reports what is inside it, including
a search for any English translation text. I start with one volume so I don't
download all eight before knowing whether they contain what I need.

Run from D:\\FYP:
    python -m cuneiform.domain3_generative.fetch_oracc_saao
    python -m cuneiform.domain3_generative.fetch_oracc_saao --volume saa10
"""

import argparse
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

BASE = "https://oracc.museum.upenn.edu"
UA = {"User-Agent": "Mozilla/5.0 (compatible; cuneiform-fyp/1.0)"}

# The eight SAAo volumes my transliterations come from.
OUR_VOLUMES = ["saa01", "saa05", "saa06", "saa08",
               "saa09", "saa10", "saa13", "saa16"]

# Words that only appear in English translation, used to find translation text.
ENGLISH_MARKERS = re.compile(
    r"\b(the king|my lord|I have|to the king|they have|his |her )",
    re.IGNORECASE)


def _fix_console_encoding() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _ssl_context():
    """
    Verify HTTPS against the Windows certificate store. The Oracc server leaves
    out an intermediate certificate, which certifi can't fill in but Windows
    can. Verification stays fully on.
    """
    if truststore is None:
        raise SystemExit(
            "truststore is not installed. Run:  pip install truststore")
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def http_get(url, timeout=180):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout,
                                context=_ssl_context()) as resp:
        return resp.read()
    

def show_manifest(volume):
    """Ask the project which JSON files it publishes before downloading."""
    url = f"{BASE}/saao/{volume}/manifest.json"
    print("-" * 68)
    print(f"1. MANIFEST for saao/{volume}")
    print("-" * 68)
    print(f"  {url}")
    try:
        data = json.loads(http_get(url, timeout=60).decode("utf-8"))
    except Exception as exc:
        print(f"  Could not fetch manifest: {exc}")
        return None
    print(f"  type    : {data.get('type')}")
    print(f"  project : {data.get('project')}")
    files = data.get("files", [])
    print(f"  files   : {len(files)}")
    for f in files[:40]:
        print(f"    {f}")
    if len(files) > 40:
        print(f"    ... and {len(files) - 40} more")
    print(f"  everything: {data.get('everything')}")
    return data


def download_volume(volume, force=False):
    name = f"saao-{volume}.zip"
    dest = config.D3_ORACC_DIR / name
    url = f"{BASE}/json/{name}"

    print()
    print("-" * 68)
    print(f"2. DOWNLOAD {name}")
    print("-" * 68)
    if dest.exists() and not force:
        print(f"  Already present ({dest.stat().st_size / 1e6:.1f} MB). "
              f"Use --force to re-download.")
        try:
            with zipfile.ZipFile(dest) as zf:
                bad = zf.testzip()
            print(f"  Archive integrity: {'OK' if bad is None else bad}")
        except zipfile.BadZipFile:
            print("  NOT A VALID ZIP -- delete it and re-download.")
            return None
        return dest
    print(f"  {url}")
    try:
        dest.write_bytes(http_get(url))
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return None
    print(f"  Saved {dest.stat().st_size / 1e6:.1f} MB to {dest}")
    return dest


def inventory(zip_path):
    print()
    print("-" * 68)
    print("3. ARCHIVE INVENTORY")
    print("-" * 68)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        print(f"  entries: {len(names):,}")

        depth1 = Counter(n.split("/")[1] if n.count("/") > 1 else n.rstrip("/")
                         for n in names)
        print("\n  Top-level structure:")
        for k, v in depth1.most_common(25):
            print(f"    {k:<28} {v:>6}")

        exts = Counter("." + n.rsplit(".", 1)[-1] if "." in n.rsplit("/", 1)[-1]
                       else "(none)" for n in names)
        print(f"\n  Extensions: {dict(exts.most_common(10))}")

        infos = sorted(zf.infolist(), key=lambda i: -i.file_size)
        print("\n  15 largest members:")
        for i in infos[:15]:
            print(f"    {i.file_size / 1e6:>8.2f} MB  {i.filename}")

        # Translations would be outside corpusjson
        others = [n for n in names
                  if "corpusjson" not in n and not n.endswith("/")]
        print(f"\n  Members OUTSIDE corpusjson/ ({len(others)}):")
        for n in others[:30]:
            print(f"    {n}")
        if len(others) > 30:
            print(f"    ... and {len(others) - 30} more")
    return names


def hunt_translations(zip_path, names):
    """Find where, if anywhere, English translation text is stored."""
    print()
    print("-" * 68)
    print("4. WHERE ARE THE ENGLISH TRANSLATIONS?")
    print("-" * 68)

    with zipfile.ZipFile(zip_path) as zf:
        # (a) any file whose name suggests translation
        by_name = [n for n in names
                   if re.search(r"(tran|eng|xtf|transl)", n, re.I)]
        print(f"  Members with translation-ish names ({len(by_name)}):")
        for n in by_name[:20]:
            print(f"    {n}")

        # (b) scan the other files for English prose
        others = [n for n in names
                  if "corpusjson" not in n and not n.endswith("/")]
        print(f"\n  Scanning {len(others)} non-corpusjson members "
              f"for English text:")
        for n in others[:25]:
            try:
                raw = zf.read(n)[:400_000].decode("utf-8", errors="replace")
            except Exception:
                continue
            hits = ENGLISH_MARKERS.findall(raw)
            if hits:
                print(f"    HIT  {n}  ({len(hits)} English markers)")
                snippet = raw[:300].replace("\n", " ")
                print(f"         {snippet[:280]}")

        # (c) look inside one corpusjson text
        cj = [n for n in names if "corpusjson" in n and n.endswith(".json")]
        print(f"\n  corpusjson text files: {len(cj):,}")
        if cj:
            sample = cj[0]
            print(f"  Inspecting {sample}")
            doc = json.loads(zf.read(sample).decode("utf-8",
                                                    errors="replace"))
            print(f"    top-level keys: {list(doc.keys())}")
            blob = json.dumps(doc)[:600_000]
            hits = ENGLISH_MARKERS.findall(blob)
            print(f"    English markers inside this text file: {len(hits)}")
            # Show any key that looks like a translation field
            found = sorted(set(re.findall(
                r'"(tran[a-z]*|eng|label|norm|sense|gw)"', blob)))
            print(f"    translation-ish JSON keys present: {found}")


def check_join(names):
    """Check that this volume's P-numbers match my corpus tablet ids."""
    import pandas as pd

    print()
    print("-" * 68)
    print("5. TABLET-ID JOIN CHECK")
    print("-" * 68)

    oracc_ids = {re.sub(r"\.json$", "", n.rsplit("/", 1)[-1])
                 for n in names if "corpusjson" in n and n.endswith(".json")}
    print(f"  text ids in this volume: {len(oracc_ids):,}")
    print(f"  sample: {sorted(oracc_ids)[:10]}")

    frames = [pd.read_csv(p)
              for p in sorted(config.TRANSLITERATIONS_DIR.glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    ours = set(df["tablet_CDLI"].astype(str).unique())

    overlap = ours & oracc_ids
    print(f"\n  our distinct tablets      : {len(ours):,}")
    print(f"  overlap with this volume  : {len(overlap):,}")
    print(f"  sample overlap            : {sorted(overlap)[:10]}")

    # And my Domain 1 held-out tablets in particular
    print(f"\n  (Full-corpus join will be measured once all eight volumes "
          f"are downloaded.)")


def main() -> None:
    _fix_console_encoding()
    config.ensure_dirs()

    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", default="saa01", choices=OUR_VOLUMES)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    print("=" * 68)
    print(f"ORACC SAAo FETCH — {args.volume}")
    print("=" * 68)

    show_manifest(args.volume)
    zip_path = download_volume(args.volume, force=args.force)
    if not zip_path:
        raise SystemExit("Download failed; nothing further to do.")

    names = inventory(zip_path)
    hunt_translations(zip_path, names)
    check_join(names)

    print()
    print("=" * 68)
    print("Section 4 above shows whether translations are in the zip")
    print("or need a different source.")
    print("=" * 68)


if __name__ == "__main__":
    main()