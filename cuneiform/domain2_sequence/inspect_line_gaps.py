"""
Finds the 110 rows build_corpus.py loaded but never emitted as tokens.
pandas.groupby() silently drops rows whose key contains NaN, so this shows
whether those rows are junk or real signs I need to keep.

Run from the project root:
    python -m cuneiform.domain2_sequence.inspect_line_gaps
"""

import pandas as pd

from cuneiform.domain2_sequence.build_corpus import (
    LINE_KEY,
    load_all_transliterations,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def main() -> None:
    df = load_all_transliterations()
    print(f"\nRows loaded: {len(df):,}")

    print("\nNull counts per column")
    print("-" * 68)
    nulls = df.isna().sum()
    for col, n in nulls.items():
        flag = "   <-- these rows are dropped by groupby" if (n and col in LINE_KEY) else ""
        print(f"  {col:<15} {n:>6,}{flag}")

    bad = df[df[LINE_KEY].isna().any(axis=1)]
    print(f"\nRows with a NaN in the line key: {len(bad):,}")

    if len(bad):
        print("\nWhich source files they come from:")
        print(bad["source_file"].value_counts().to_string())

        print("\nWhat sign labels they carry:")
        print(bad["train_label"].value_counts().head(10).to_string())

        print("\nFirst 15 offending rows:")
        print(bad.head(15).to_string(index=False))
    else:
        print("\nNo NaNs in the line key. The loss came from somewhere else --")
        print("tell me and I will dig further.")

    kept = df.dropna(subset=LINE_KEY)
    print("\n" + "=" * 68)
    print(f"  rows in  : {len(df):,}")
    print(f"  rows out : {len(kept):,}")
    print(f"  lost     : {len(df) - len(kept):,}")
    print("=" * 68)


if __name__ == "__main__":
    main()