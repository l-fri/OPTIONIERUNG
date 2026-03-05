from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


TRANSACTION_SECTION = "Transaction History"

OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")

EU_RE = re.compile(
    r"^(?P<cp>[PC])\s+(?P<und>[A-Z0-9.]+)\s+(?P<yyyymmdd>\d{8})\s+(?P<strike>\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)


def expand_paths(single_path_or_glob: str) -> list[Path]:
    """Resolve --csv path (can be a single file or a glob)."""
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


@dataclass(frozen=True)
class OptionMeta:
    underlying: str | None
    cp: str | None


def parse_option_symbol(symbol: str) -> OptionMeta | None:
    if symbol is None:
        return None
    s = str(symbol).strip()
    if s in {"", "-"}:
        return None

    key = re.sub(r"\s+", "", s).upper()
    m = OCC_RE.match(key)
    if m:
        und, _yymmdd, cp, _strike8 = m.groups()
        return OptionMeta(underlying=und, cp=cp)

    m2 = EU_RE.match(s.upper())
    if m2:
        return OptionMeta(underlying=m2.group("und").upper(), cp=m2.group("cp").upper())

    return None


def to_float(x) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if s in {"", "-", "nan", "NaN"}:
        return None
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


def _add_value_labels(ax, bars):
    for b in bars:
        h = b.get_height()
        x = b.get_x() + b.get_width() / 2
        if abs(h) < 1e-12:
            continue
        label = eur_fmt(h)
        offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015
        y = h + offset if h > 0 else h - offset
        va = "bottom" if h > 0 else "top"
        ax.text(x, y, label, ha="center", va=va, fontsize=8, rotation=0)


def plot_monthly_bars(monthly: pd.DataFrame, out_path: Path) -> None:
    months = monthly["MonthLabel"].tolist()
    x = np.arange(len(months))
    w = 0.20

    cashflow = monthly["NetCashflow"].to_numpy()
    received = monthly["NetReceived"].to_numpy()
    paid = monthly["NetPaid"].to_numpy()
    commission = monthly["CommissionPaid"].to_numpy()

    fig, ax = plt.subplots(figsize=(max(10, len(months) * 1.1), 6))


    cashflow_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in cashflow]

    b1 = ax.bar(
        x - 1.5 * w,
        cashflow,
        width=w,
        label="Netto Cashflow",
        color=cashflow_colors,
    )

    b2 = ax.bar(
        x - 0.5 * w,
        received,
        width=w,
        label="Netto Prämie erhalten",
        color="#1f77b4",
    )

    b3 = ax.bar(
        x + 0.5 * w,
        paid,
        width=w,
        label="Netto Prämie gezahlt",
        color="#ff7f0e",
    )

    b4 = ax.bar(
        x + 1.5 * w,
        commission,
        width=w,
        label="Provision netto bezahlt",
        color="#000000",
    )

    ax.axhline(0, linewidth=1)
    ax.set_title("Options: Monatliche Prämienübersicht (EUR)")
    ax.set_xlabel("Monat")
    ax.set_ylabel("EUR")
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.legend()

    y_min = min(0.0, float(np.min([cashflow, -paid, -commission]) if len(months) else 0.0))
    y_max = float(np.max([cashflow, received, paid, commission]) if len(months) else 0.0)
    pad = max(10.0, (y_max - y_min) * 0.12)
    ax.set_ylim(y_min - pad, y_max + pad)

    _add_value_labels(ax, b1)
    _add_value_labels(ax, b2)
    _add_value_labels(ax, b3)
    _add_value_labels(ax, b4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Summiert Optionsprämien (Cashflows) aus LYNX/IBKR Transaction History CSV in EUR und erstellt ein Monats-Balkendiagramm."
    )
    ap.add_argument("--csv", required=True, help="CSV Pfad (oder Glob), z.B. --csv data/U24066232.TRANSACTIONS.YTD.csv")
    args = ap.parse_args()

    paths = expand_paths(args.csv)
    df = read_transaction_history(paths)

    required = {"Date", "Transaction Type", "Symbol", "Gross Amount", "Commission", "Net Amount"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Spalten fehlen: {missing}. Gefunden: {list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Transaction Type"] = df["Transaction Type"].astype(str)
    df["Symbol"] = df["Symbol"].astype(str)

    for col in ["Gross Amount", "Commission", "Net Amount"]:
        df[col] = df[col].map(to_float)

    df["OptMeta"] = df["Symbol"].map(parse_option_symbol)
    opt = df[df["OptMeta"].notna() & df["Transaction Type"].isin(["Buy", "Sell"])].copy()

    if opt.empty:
        print("Keine Option-Trades gefunden (Buy/Sell + optionsähnliches Symbol).")
        return 0

    opt["NetCents"] = opt["Net Amount"].fillna(0.0).map(to_cents)
    opt["GrossCents"] = opt["Gross Amount"].fillna(0.0).map(to_cents)
    opt["CommCents"] = opt["Commission"].fillna(0.0).map(to_cents)

    net_cents = opt["NetCents"]
    gross_cents = opt["GrossCents"]
    comm_cents = opt["CommCents"]

    net_received = net_cents[net_cents > 0].sum() / 100
    net_paid = (-net_cents[net_cents < 0].sum()) / 100
    net_total = net_cents.sum() / 100

    gross_received = gross_cents[gross_cents > 0].sum() / 100
    gross_paid = (-gross_cents[gross_cents < 0].sum()) / 100
    commission_total = comm_cents.sum() / 100  # typically negative

    dmin = opt["Date"].min()
    dmax = opt["Date"].max()

    rows = [
        ["Zeitraum", f"{dmin.date()} … {dmax.date()}"],
        ["Anzahl Option-Trades (Buy/Sell)", f"{int(len(opt))}"],
        ["Netto Prämien erhalten (EUR)", eur_fmt(net_received)],
        ["Netto Prämien gezahlt (EUR)", eur_fmt(net_paid)],
        ["Netto Summe (EUR)", eur_fmt(net_total)],
        ["Brutto erhalten (EUR)", eur_fmt(gross_received)],
        ["Brutto gezahlt (EUR)", eur_fmt(gross_paid)],
        ["Provision Summe (EUR)", eur_fmt(commission_total)],
    ]
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="github"))

    opt["Month"] = opt["Date"].dt.to_period("M").dt.to_timestamp()

    monthly = opt.groupby("Month", as_index=False).agg(
        NetCashflowCents=("NetCents", "sum"),
        NetReceivedCents=("NetCents", lambda s: int(s[s > 0].sum())),
        NetPaidCents=("NetCents", lambda s: int((-s[s < 0].sum()))),
        CommSumCents=("CommCents", "sum"),
    )
    monthly = monthly.sort_values("Month").reset_index(drop=True)

    monthly["MonthLabel"] = monthly["Month"].dt.strftime("%Y-%m")
    monthly["NetCashflow"] = monthly["NetCashflowCents"] / 100.0
    monthly["NetReceived"] = monthly["NetReceivedCents"] / 100.0
    monthly["NetPaid"] = monthly["NetPaidCents"] / 100.0
    monthly["CommissionPaid"] = (-monthly["CommSumCents"]) / 100.0

    out_path = Path("options_diagramm.png")
    plot_monthly_bars(
        monthly[["MonthLabel", "NetCashflow", "NetReceived", "NetPaid", "CommissionPaid"]],
        out_path=out_path,
    )
    print(f"\nDiagramm gespeichert: {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())