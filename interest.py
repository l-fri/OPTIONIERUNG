from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate


TRANSACTION_SECTION = "Transaction History"


def expand_paths(single_path_or_glob: str) -> list[Path]:
    g = single_path_or_glob.strip()
    if any(ch in g for ch in ["*", "?", "["]):
        out = sorted(Path().glob(g))
    else:
        out = [Path(g)]
    out = [p for p in out if p.exists()]
    if not out:
        raise SystemExit("Keine CSV gefunden.")
    return out


def read_transaction_history(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for p in paths:
        header: list[str] | None = None
        rows: list[list[str]] = []

        with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            for r in reader:
                if not r or r[0] != TRANSACTION_SECTION:
                    continue
                if len(r) < 3:
                    continue
                kind = (r[1] or "").strip()
                if kind == "Header":
                    header = [c.strip() for c in r[2:]]
                elif kind == "Data":
                    if header is None:
                        raise ValueError(f"Header fehlt vor Daten in {p}")
                    rows.append(r[2:])

        if header is None:
            raise ValueError(f"Keine '{TRANSACTION_SECTION}' Sektion in {p}")

        df = pd.DataFrame(rows, columns=header)
        df["SourceFile"] = str(p)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out.columns = [c.strip() for c in out.columns]
    return out


def to_float(x) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s in {"", "-", "nan", "NaN"}:
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def to_cents(x: float | int | None) -> int:
    if x is None:
        return 0
    return int(round(float(x) * 100))


def eur_fmt(x: float | int) -> str:
    s = f"{float(x):,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s


def add_value_labels(ax, bars):
    y0, y1 = ax.get_ylim()
    offset = (y1 - y0) * 0.015
    for b in bars:
        h = b.get_height()
        if abs(h) < 1e-12:
            continue
        x = b.get_x() + b.get_width() / 2
        y = h + offset if h > 0 else h - offset
        va = "bottom" if h > 0 else "top"
        ax.text(x, y, eur_fmt(h), ha="center", va=va, fontsize=8)


def plot_monthly_interest(monthly: pd.DataFrame, out_path: Path) -> None:
    months = monthly["MonthLabel"].tolist()
    x = np.arange(len(months))
    y = monthly["InterestNet"].to_numpy(dtype=float)

    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in y]

    fig, ax = plt.subplots(figsize=(max(10, len(months) * 1.05), 6))
    bars = ax.bar(x, y, color=colors)

    ax.axhline(0, linewidth=1)
    ax.set_title("Zinsen (Soll/Haben): Netto pro Monat (EUR)")
    ax.set_xlabel("Monat")
    ax.set_ylabel("EUR")
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right")

    y_min = float(min(0.0, y.min(initial=0.0)))
    y_max = float(max(0.0, y.max(initial=0.0)))
    pad = max(10.0, (y_max - y_min) * 0.20)
    ax.set_ylim(y_min - pad, y_max + pad)

    add_value_labels(ax, bars)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Summiert Netto-Zinsen (Sollzinsen/Habenzinsen) aus LYNX/IBKR Transaction History CSV (EUR) + Monatsdiagramm."
    )
    ap.add_argument("--csv", required=True, help="CSV Pfad (oder Glob), z.B. --csv data/U24066232.TRANSACTIONS.YTD.csv")
    args = ap.parse_args()

    paths = expand_paths(args.csv)
    df = read_transaction_history(paths)

    required = {"Date", "Transaction Type", "Net Amount"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Spalten fehlen: {missing}. Gefunden: {list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Transaction Type"] = df["Transaction Type"].astype(str)
    df["Net Amount"] = df["Net Amount"].map(to_float)

    interest = df[df["Transaction Type"].isin(["Debit Interest", "Credit Interest"])].copy()

    if interest.empty:
        interest = df[df["Transaction Type"].str.contains("Interest", case=False, na=False)].copy()

    if interest.empty:
        print("Keine Zins-Zeilen gefunden (Debit/Credit Interest).")
        return 0

    interest["NetCents"] = interest["Net Amount"].fillna(0.0).map(to_cents)

    total_net = interest["NetCents"].sum() / 100.0
    total_paid = (-interest.loc[interest["NetCents"] < 0, "NetCents"].sum()) / 100.0
    total_received = (interest.loc[interest["NetCents"] > 0, "NetCents"].sum()) / 100.0

    dmin = interest["Date"].min()
    dmax = interest["Date"].max()

    rows = [
        ["Zeitraum", f"{dmin.date()} … {dmax.date()}" if pd.notna(dmin) else "—"],
        ["Netto Zinsen gesamt (EUR, signed)", eur_fmt(total_net)],
        ["Davon gezahlt (EUR)", eur_fmt(total_paid)],
        ["Davon erhalten (EUR)", eur_fmt(total_received)],
    ]
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="github"))

    interest["Month"] = interest["Date"].dt.to_period("M").dt.to_timestamp()
    monthly = interest.groupby("Month", as_index=False)["NetCents"].sum().sort_values("Month").reset_index(drop=True)
    monthly["MonthLabel"] = monthly["Month"].dt.strftime("%Y-%m")
    monthly["InterestNet"] = monthly["NetCents"] / 100.0

    out_path = Path("interest_diagramm.png")
    plot_monthly_interest(monthly[["MonthLabel", "InterestNet"]], out_path=out_path)
    print(f"\nDiagramm gespeichert: {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())