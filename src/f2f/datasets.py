"""Stream 1: load SocioPatterns contact networks into one contact table.

Every dataset becomes the same two tables:

contacts : DataFrame with columns
    t        int   seconds since the start of the recording (first contact at t >= 0)
    i, j     int   node ids (0..N-1, the Netzschleuder node index), always i < j
    abs_time datetime (Europe/Paris) absolute time of the contact window end
    day      int   calendar-day index, 0 for the first day of the recording
nodes : DataFrame indexed by node id with columns
    group    str   the metadata partition used everywhere (ward role / class)
    plus any other metadata column the file has (e.g. gender)

Contacts are 20-second windows: a row (t, i, j) means i and j were in face-to-face
proximity during [t - 20 s, t]. Two rows for the same pair at consecutive t are two
windows of one longer contact.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
WINDOW_S = 20  # sampling resolution of the SocioPatterns sensors, in seconds


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    folder: str
    group_col: str  # metadata column that defines the partition (status / class)
    t0: pd.Timestamp  # absolute time at which the raw file's t = 0
    n_nodes: int  # expected counts, used as checksums
    n_contacts: int
    citation: str
    url: str


DATASETS: dict[str, DatasetSpec] = {
    "hospital": DatasetSpec(
        key="hospital",
        folder="sp_hospital",
        group_col="status",
        # The file says: from Monday 6 Dec 2010 13:00; t is seconds since that start.
        t0=pd.Timestamp("2010-12-06 13:00", tz="Europe/Paris"),
        n_nodes=75,
        n_contacts=32424,
        citation="Vanhems et al., PLoS ONE 8(9) e73970 (2013), doi:10.1371/journal.pone.0073970",
        url="https://networks.skewed.de/net/sp_hospital",
    ),
    "primary_school": DatasetSpec(
        key="primary_school",
        folder="sp_primary_school",
        group_col="class",
        # t is seconds since midnight of the first day (first contact at 08:40, day 2 at ~ 86400 + 8 h).
        t0=pd.Timestamp("2009-10-01 00:00", tz="Europe/Paris"),
        n_nodes=242,
        n_contacts=125773,
        citation="Stehle et al., PLoS ONE 6(8) e23176 (2011), doi:10.1371/journal.pone.0023176",
        url="https://networks.skewed.de/net/sp_primary_school",
    ),
    "high_school": DatasetSpec(
        key="high_school",
        folder="sp_high_school_proximity",
        group_col="class",
        # t is UNIX time (seconds since 1970-01-01 UTC).
        t0=pd.Timestamp("1970-01-01 00:00", tz="UTC"),
        n_nodes=329,
        n_contacts=188508,
        citation="Mastrandrea et al., PLoS ONE 10(9) e0136497 (2015), doi:10.1371/journal.pone.0136497",
        url="https://networks.skewed.de/net/sp_high_school",
    ),
}


def _read_netzschleuder_csv(path: Path) -> pd.DataFrame:
    """Read a Netzschleuder CSV whose first line is a '# col, col, ...' header."""
    with open(path) as fh:
        header = fh.readline()
    if not header.startswith("#"):
        raise ValueError(f"{path}: expected a '# col, col' header line")
    cols = [c.strip() for c in header.lstrip("#").split(",")]
    return pd.read_csv(path, comment="#", header=None, names=cols)


def raw_available(key: str) -> bool:
    spec = DATASETS[key]
    folder = RAW_DIR / spec.folder
    return (folder / "edges.csv").exists() and (folder / "nodes.csv").exists()


def load_nodes(key: str) -> pd.DataFrame:
    spec = DATASETS[key]
    nodes = _read_netzschleuder_csv(RAW_DIR / spec.folder / "nodes.csv")
    nodes = nodes.drop(columns=[c for c in nodes.columns if c.startswith("_")])
    nodes = nodes.rename(columns={"index": "node", "id": "sensor_id", spec.group_col: "group"})
    nodes["group"] = nodes["group"].astype(str)
    return nodes.set_index("node").sort_index()


def load_contacts(key: str, check: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (contacts, nodes) for one dataset. See the module docstring for the schema."""
    spec = DATASETS[key]
    if not raw_available(key):
        raise FileNotFoundError(
            f"Raw files for '{key}' not found under {RAW_DIR / spec.folder}. "
            "Run scripts/download_data.sh first."
        )
    nodes = load_nodes(key)
    edges = _read_netzschleuder_csv(RAW_DIR / spec.folder / "edges.csv")
    edges = edges.rename(columns={"source": "i", "target": "j", "time": "t_raw"})

    i = np.minimum(edges["i"].to_numpy(), edges["j"].to_numpy())
    j = np.maximum(edges["i"].to_numpy(), edges["j"].to_numpy())
    contacts = pd.DataFrame({"t_raw": edges["t_raw"].astype(np.int64), "i": i, "j": j})
    contacts = contacts[contacts["i"] != contacts["j"]].drop_duplicates()

    abs_time = spec.t0 + pd.to_timedelta(contacts["t_raw"], unit="s")
    abs_time = abs_time.dt.tz_convert("Europe/Paris")
    day_date = abs_time.dt.floor("D")
    first_day = day_date.min()
    contacts["t"] = contacts["t_raw"] - contacts["t_raw"].min()
    contacts["abs_time"] = abs_time
    contacts["day"] = ((day_date - first_day) / pd.Timedelta(days=1)).astype(int)

    contacts = (
        contacts[["t", "i", "j", "abs_time", "day"]]
        .sort_values(["t", "i", "j"])
        .reset_index(drop=True)
    )

    if check:
        checksum(key, contacts, nodes)
    return contacts, nodes


def checksum(key: str, contacts: pd.DataFrame, nodes: pd.DataFrame) -> None:
    """Raise if node or contact counts differ from the published dataset sizes."""
    spec = DATASETS[key]
    active = pd.unique(np.concatenate([contacts["i"].to_numpy(), contacts["j"].to_numpy()]))
    problems = []
    if len(nodes) != spec.n_nodes:
        problems.append(f"nodes: {len(nodes)} != {spec.n_nodes}")
    if len(contacts) != spec.n_contacts:
        problems.append(f"contacts: {len(contacts)} != {spec.n_contacts}")
    if not set(active).issubset(set(nodes.index)):
        problems.append("contacts reference node ids missing from the node table")
    if (contacts["t"] % WINDOW_S).any():
        problems.append(f"timestamps are not multiples of {WINDOW_S} s")
    if problems:
        raise AssertionError(f"checksum failed for '{key}': " + "; ".join(problems))


def describe(key: str) -> dict:
    """Small summary used by scripts/check_data.py."""
    contacts, nodes = load_contacts(key)
    active = pd.unique(np.concatenate([contacts["i"].to_numpy(), contacts["j"].to_numpy()]))
    per_day = contacts.groupby("day").agg(
        contacts=("t", "size"),
        start=("abs_time", "min"),
        end=("abs_time", "max"),
        active_nodes=("i", lambda s: 0),  # placeholder, filled below
    )
    per_day["active_nodes"] = [
        len(pd.unique(np.concatenate([g["i"].to_numpy(), g["j"].to_numpy()])))
        for _, g in contacts.groupby("day")
    ]
    return {
        "key": key,
        "nodes": len(nodes),
        "active_nodes": len(active),
        "contacts": len(contacts),
        "span_hours": contacts["t"].max() / 3600,
        "days": int(contacts["day"].nunique()),
        "groups": nodes["group"].value_counts().to_dict(),
        "per_day": per_day,
    }
