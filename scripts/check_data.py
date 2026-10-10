"""Stream 1 acceptance check: load every dataset, verify checksums, print the day table.

Usage:  python scripts/check_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from f2f import datasets, views  # noqa: E402

pd.set_option("display.width", 160)


def main() -> int:
    ok = True
    for key in datasets.DATASETS:
        print(f"\n=== {key}  ({datasets.DATASETS[key].citation})")
        if not datasets.raw_available(key):
            print("   raw files missing: run scripts/download_data.sh")
            ok = False
            continue
        try:
            contacts, nodes = datasets.load_contacts(key)
        except AssertionError as e:
            print("   CHECKSUM FAILED:", e)
            ok = False
            continue
        G = views.aggregate(contacts, nodes)
        snaps5 = views.to_snapshots(contacts, 300)
        print(f"   nodes {len(nodes)}  contacts {len(contacts)}  span {contacts['t'].max()/3600:.1f} h  "
              f"days {contacts['day'].nunique()}  aggregated links {G.number_of_edges()}  "
              f"5-min steps {snaps5.attrs['n_steps']}  5-min pair-steps {len(snaps5)}")
        print("   groups:", nodes["group"].value_counts().to_dict())
        dt = views.day_table(contacts)
        dt["abs_start"] = dt["abs_start"].dt.strftime("%a %d %b %H:%M")
        dt["abs_end"] = dt["abs_end"].dt.strftime("%a %d %b %H:%M")
        print(dt.to_string())
        cal, val = views.split_days(contacts, 1)
        print(f"   split: calibration day 0 -> {len(cal)} contacts; validation -> {len(val)} contacts")
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
