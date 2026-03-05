from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate


TRANSACTION_SECTION = "Transaction History"
FX_PAIR_RE = re.compile(r"^[A-Z]{3}\.[A-Z]{3}$")


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


def is_fx_pair(sym: str) -> bool:
    return bool(FX_PAIR_RE.match(str(sym).strip().upper()))


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


def plot_monthly_cash_fx(monthly: pd.DataFrame, out_path: Path) -> None:
    months = monthly["MonthLabel"].tolist()
    x = np.arange(len(months))
    w = 0.17

    deposits = monthly["Deposits"].to_numpy(dtype=float)
    withdrawals = monthly["Withdrawals"].to_numpy(dtype=float)
    eur_to_usd = monthly["EurToUsd"].to_numpy(dtype=float)
    fees = monthly["FxFeesPaid"].to_numpy(dtype=float)
    spread = monthly["FxSpreadPaid"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(10, len(months) * 1.15), 6))


    b1 = ax.bar(x - 2 * w, deposits, width=w, label="Einzahlungen", color="#2ca02c")
    b2 = ax.bar(x - 1 * w, withdrawals, width=w, label="Auszahlungen", color="#d62728")
    b3 = ax.bar(x + 0 * w, eur_to_usd, width=w, label="EUR → USD umgerechnet", color="#1f77b4")
    b4 = ax.bar(x + 1 * w, fees, width=w, label="FX Provision (bezahlt)", color="#000000")
    b5 = ax.bar(x + 2 * w, spread, width=w, label="FX Spread/Execution (bezahlt)", color="#7f7f7f")

    ax.axhline(0, linewidth=1)
    ax.set_title("Cash & FX: Einzahlungen / Umrechnung / FX-Kosten pro Monat (EUR)")
    ax.set_xlabel("Monat")
    ax.set_ylabel("EUR")
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.legend(ncols=2)

    y_min = float(min(0.0, deposits.min(initial=0.0)))
    y_max = float(
        max(
            0.0,
            deposits.max(initial=0.0),
            withdrawals.max(initial=0.0),
            eur_to_usd.max(initial=0.0),
            fees.max(initial=0.0),
            spread.max(initial=0.0),
        )
    )
    pad = max(10.0, (y_max - y_min) * 0.18)
    ax.set_ylim(y_min - pad, y_max + pad)

    add_value_labels(ax, b1)
    add_value_labels(ax, b2)
    add_value_labels(ax, b3)
    add_value_labels(ax, b4)
    add_value_labels(ax, b5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Zeigt Einzahlungen/Auszahlungen, EUR→USD Umrechnung und FX-Kosten (Provision + NetAmount/Spread) aus LYNX/IBKR CSV (EUR) + Monatsdiagramm."
    )
    ap.add_argument("--csv", required=True, help="CSV Pfad (oder Glob), z.B. --csv data/U24066232.TRANSACTIONS.YTD.csv")
    args = ap.parse_args()

    paths = expand_paths(args.csv)
    df = read_transaction_history(paths)

    required = {"Date", "Transaction Type", "Symbol", "Quantity", "Net Amount", "Commission"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Spalten fehlen: {missing}. Gefunden: {list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Transaction Type"] = df["Transaction Type"].astype(str)
    df["Symbol"] = df["Symbol"].astype(str)

    df["Quantity"] = df["Quantity"].map(to_float)
    df["Net Amount"] = df["Net Amount"].map(to_float)
    df["Commission"] = df["Commission"].map(to_float)

    dep = df[df["Transaction Type"].eq("Deposit")].copy()
    wdr = df[df["Transaction Type"].eq("Withdraw")].copy()

    dep_cents = dep["Net Amount"].fillna(0.0).map(to_cents)
    wdr_cents = wdr["Net Amount"].fillna(0.0).map(to_cents)

    deposits_total = dep_cents.sum() / 100.0
    withdrawals_total = (-wdr_cents[wdr_cents < 0].sum()) / 100.0
    net_cash_in = (dep_cents.sum() + wdr_cents.sum()) / 100.0

    df["IsFx"] = df["Transaction Type"].str.contains("Forex", case=False, na=False) | df["Symbol"].map(is_fx_pair)
    fx = df[df["IsFx"]].copy()

    fx_net_cents = fx["Net Amount"].fillna(0.0).map(to_cents)
    fx_comm_cents = fx["Commission"].fillna(0.0).map(to_cents)

    fx_net_total = fx_net_cents.sum() / 100.0
    fx_comm_total = fx_comm_cents.sum() / 100.0
    fx_fees_paid = (-fx_comm_cents.sum()) / 100.0
    fx_spread_paid = (-fx_net_cents.sum()) / 100.0
    fx_total_cost = fx_net_total + fx_comm_total

    eurusd = df[df["Symbol"].str.strip().str.upper().eq("EUR.USD")].copy()
    eurusd_qty = eurusd["Quantity"].fillna(0.0)

    eur_to_usd_total = float((-eurusd_qty[eurusd_qty < 0].sum()))
    eur_from_usd_total = float((eurusd_qty[eurusd_qty > 0].sum()))
    eur_net_converted = eur_to_usd_total - eur_from_usd_total

    dmin = df["Date"].min()
    dmax = df["Date"].max()

    rows = [
        ["Zeitraum", f"{dmin.date()} … {dmax.date()}" if pd.notna(dmin) else "—"],
        ["Einzahlungen gesamt (EUR)", eur_fmt(deposits_total)],
        ["Auszahlungen gesamt (EUR)", eur_fmt(withdrawals_total)],
        ["Netto Cash In (EUR)", eur_fmt(net_cash_in)],
        ["EUR → USD umgerechnet (EUR-Betrag)", eur_fmt(eur_to_usd_total)],
        ["USD → EUR umgerechnet (EUR-Betrag)", eur_fmt(eur_from_usd_total)],
        ["Netto in USD gewechselt (EUR)", eur_fmt(eur_net_converted)],
        ["FX Net Amount Summe (EUR, signed)", eur_fmt(fx_net_total)],
        ["FX Gebühren/Provision (EUR, signed)", eur_fmt(fx_comm_total)],
        ["FX Gesamtkosten (EUR, signed)", eur_fmt(fx_total_cost)],
        ["FX Provision bezahlt (EUR)", eur_fmt(fx_fees_paid)],
        ["FX Spread/Execution bezahlt (EUR)", eur_fmt(fx_spread_paid)],
    ]
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="github"))

    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    dep["Month"] = dep["Date"].dt.to_period("M").dt.to_timestamp()
    wdr["Month"] = wdr["Date"].dt.to_period("M").dt.to_timestamp()
    eurusd["Month"] = eurusd["Date"].dt.to_period("M").dt.to_timestamp()
    fx["Month"] = fx["Date"].dt.to_period("M").dt.to_timestamp()

    dep_m = dep.groupby("Month")["Net Amount"].sum().rename("Deposits").reset_index()
    wdr_m = (-wdr.groupby("Month")["Net Amount"].sum()).rename("Withdrawals").reset_index()

    eur_to_usd_m = (
        eurusd.assign(Q=eurusd["Quantity"].fillna(0.0))
        .groupby("Month")["Q"]
        .apply(lambda s: float((-s[s < 0].sum())))
        .rename("EurToUsd")
        .reset_index()
    )

    fx_fees_m = (
        fx.groupby("Month")["Commission"]
        .sum()
        .fillna(0.0)
        .apply(lambda v: float(-v))
        .rename("FxFeesPaid")
        .reset_index()
    )

    fx_spread_m = (
        fx.groupby("Month")["Net Amount"]
        .sum()
        .fillna(0.0)
        .apply(lambda v: float(-v))
        .rename("FxSpreadPaid")
        .reset_index()
    )

    months = pd.Series(pd.to_datetime(df["Month"].dropna().unique())).sort_values()
    monthly = pd.DataFrame({"Month": months}).drop_duplicates().sort_values("Month").reset_index(drop=True)

    monthly = monthly.merge(dep_m, on="Month", how="left")
    monthly = monthly.merge(wdr_m, on="Month", how="left")
    monthly = monthly.merge(eur_to_usd_m, on="Month", how="left")
    monthly = monthly.merge(fx_fees_m, on="Month", how="left")
    monthly = monthly.merge(fx_spread_m, on="Month", how="left")

    monthly = monthly.fillna(0.0)
    monthly["MonthLabel"] = monthly["Month"].dt.strftime("%Y-%m")

    out_path = Path("cash_fx_diagramm.png")
    plot_monthly_cash_fx(
        monthly[["MonthLabel", "Deposits", "Withdrawals", "EurToUsd", "FxFeesPaid", "FxSpreadPaid"]],
        out_path=out_path,
    )
    print(f"\nDiagramm gespeichert: {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())