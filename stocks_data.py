from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate


TRANSACTION_SECTION = "Transaction History"

OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")

EU_RE = re.compile(
    r"^(?P<cp>[PC])\s+(?P<und>[A-Z0-9.]+)\s+(?P<yyyymmdd>\d{8})\s+(?P<strike>\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)


def expand_paths(globs: list[str]) -> list[Path]:
    if not globs:
        globs = ["data/*.csv"]
    out: list[Path] = []
    for g in globs:
        if any(ch in g for ch in ["*", "?", "["]):
            out.extend(sorted(Path().glob(g)))
        else:
            out.append(Path(g))
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


def parse_option_symbol(symbol: str) -> bool:
    if symbol is None:
        return False
    s = str(symbol).strip()
    if s in {"", "-"}:
        return False

    key = re.sub(r"\s+", "", s).upper()
    if OCC_RE.match(key):
        return True
    if EU_RE.match(s.upper()):
        return True
    return False


def to_float(x) -> Optional[float]:
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


def eur_fmt(x_eur: float | int) -> str:
    s = f"{float(x_eur):,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s


def sign(x: float) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


@dataclass
class Lot:
    open_dt: pd.Timestamp
    open_seq: int
    open_qty: float
    rest_qty: float
    remaining_open_cents: int


@dataclass
class Match:
    symbol: str
    close_dt: pd.Timestamp
    close_seq: int
    match_qty_abs: float
    open_alloc_cents: int
    close_alloc_cents: int
    realized_cents: int


def fifo_realized_stocks(trades: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = trades.copy()
    df = df.sort_values(["Date", "Seq"], kind="mergesort").reset_index(drop=True)
    df["RealizedStockEUR"] = 0.0

    lots: Dict[str, List[Lot]] = {}

    for idx, r in df.iterrows():
        sym = str(r["Symbol"]).strip().upper()
        dt = r["Date"]
        seq = int(r["Seq"])
        qty = float(r["Quantity"])
        net_cents_total = to_cents(r["Net Amount"])

        if sym not in lots:
            lots[sym] = []

        q_remaining = qty
        trade_remaining_qty_abs = abs(qty)
        trade_remaining_cents = net_cents_total

        lot_list = lots[sym]
        lot_list.sort(key=lambda L: (L.open_dt, L.open_seq))

        realized_cents_this_trade = 0

        i = 0
        while i < len(lot_list) and sign(q_remaining) != 0:
            lot = lot_list[i]
            if sign(lot.rest_qty) == 0:
                i += 1
                continue

            if sign(lot.rest_qty) != -sign(q_remaining):
                i += 1
                continue

            match_qty = min(abs(q_remaining), abs(lot.rest_qty))
            if match_qty <= 0:
                i += 1
                continue


            lot_remaining_qty_abs = abs(lot.rest_qty)
            if lot_remaining_qty_abs <= 0:
                i += 1
                continue

            open_alloc = int(round(lot.remaining_open_cents * (match_qty / lot_remaining_qty_abs)))

            close_alloc = (
                int(round(trade_remaining_cents * (match_qty / trade_remaining_qty_abs)))
                if trade_remaining_qty_abs > 0
                else 0
            )

            realized_part = open_alloc + close_alloc
            realized_cents_this_trade += realized_part

            lot.rest_qty = lot.rest_qty + (-match_qty if lot.rest_qty > 0 else +match_qty)
            lot.remaining_open_cents -= open_alloc

            q_remaining = q_remaining + (-match_qty if q_remaining > 0 else +match_qty)
            trade_remaining_qty_abs -= match_qty
            trade_remaining_cents -= close_alloc

            if abs(lot.rest_qty) < 1e-12:
                lot.rest_qty = 0.0
                lot.remaining_open_cents = 0

            if abs(q_remaining) < 1e-12:
                q_remaining = 0.0

            i += 1

        if sign(q_remaining) != 0:
            lot_list.append(
                Lot(
                    open_dt=dt,
                    open_seq=seq,
                    open_qty=q_remaining,
                    rest_qty=q_remaining,
                    remaining_open_cents=trade_remaining_cents,
                )
            )

        df.at[idx, "RealizedStockEUR"] = realized_cents_this_trade / 100.0

    lot_rows = []
    for sym, lot_list in lots.items():
        for lot in lot_list:
            if abs(lot.rest_qty) < 1e-12:
                continue
            lot_rows.append(
                {
                    "Symbol": sym,
                    "OpenDate": lot.open_dt,
                    "OpenSeq": lot.open_seq,
                    "OpenQty": lot.open_qty,
                    "RestQty": lot.rest_qty,
                    "OpenCashflowEUR_Remaining": lot.remaining_open_cents / 100.0,
                }
            )
    lots_df = pd.DataFrame(lot_rows)
    if not lots_df.empty:
        lots_df = lots_df.sort_values(["Symbol", "OpenDate", "OpenSeq"]).reset_index(drop=True)

    return df, lots_df


def plot_monthly_realized_pnl(out: pd.DataFrame, out_path: Path) -> None:
    if out.empty:
        return

    d = out.copy()
    d["Month"] = d["Date"].dt.to_period("M").dt.to_timestamp()
    monthly = d.groupby("Month", as_index=False)["RealizedStockEUR"].sum()
    monthly = monthly.sort_values("Month").reset_index(drop=True)
    monthly["MonthLabel"] = monthly["Month"].dt.strftime("%Y-%m")

    x = np.arange(len(monthly))
    y = monthly["RealizedStockEUR"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(10, len(monthly) * 1.05), 6))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in y]
    bars = ax.bar(x, y, color=colors)

    ax.axhline(0, linewidth=1)
    ax.set_title("Aktien: Realisierte FIFO P&L pro Monat (EUR)")
    ax.set_xlabel("Monat")
    ax.set_ylabel("EUR")
    ax.set_xticks(x)
    ax.set_xticklabels(monthly["MonthLabel"].tolist(), rotation=45, ha="right")

    y0, y1 = ax.get_ylim()
    span = max(1.0, y1 - y0)
    offset = span * 0.015

    y_min = float(min(0.0, y.min(initial=0.0)))
    y_max = float(max(0.0, y.max(initial=0.0)))
    pad = max(10.0, (y_max - y_min) * 0.20)
    ax.set_ylim(y_min - pad, y_max + pad)

    for b in bars:
        h = b.get_height()
        if abs(h) < 1e-12:
            continue
        label = eur_fmt(h)
        x_text = b.get_x() + b.get_width() / 2
        y_text = h + offset if h > 0 else h - offset
        va = "bottom" if h > 0 else "top"
        ax.text(x_text, y_text, label, ha="center", va=va, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.show()
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Realisierte Aktien-P&L (FIFO) aus LYNX/IBKR Transaction History CSV in EUR.")
    ap.add_argument("--csv", action="append", default=[], help="CSV Pfad oder Glob (repeatable), z.B. --csv data/*.csv")

    ap.add_argument("--year", type=int, default=None, help="Ausgabe-Filter: Kalenderjahr (z.B. 2026).")
    ap.add_argument("--from", dest="date_from", default=None, help="Ausgabe-Filter: Startdatum YYYY-MM-DD")
    ap.add_argument("--to", dest="date_to", default=None, help="As-of: nutze Trades nur bis zu diesem Datum (YYYY-MM-DD).")

    ap.add_argument("--by-symbol", action="store_true", help="Breakdown der realisierten P&L nach Symbol ausgeben.")
    ap.add_argument("--top", type=int, default=20, help="Top-N Symbole im Breakdown (Default 20).")
    ap.add_argument("--show-open", action="store_true", help="Offene Positionen (Lots) anzeigen.")
    ap.add_argument("--open-limit", type=int, default=50, help="Limit für offene Lots-Ausgabe.")

    args = ap.parse_args()

    paths = expand_paths(args.csv)
    df = read_transaction_history(paths)

    required = {"Date", "Transaction Type", "Symbol", "Quantity", "Net Amount", "Gross Amount", "Commission"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Spalten fehlen: {missing}. Gefunden: {list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Transaction Type"] = df["Transaction Type"].astype(str)
    df["Symbol"] = df["Symbol"].astype(str)

    for col in ["Quantity", "Net Amount", "Gross Amount", "Commission"]:
        df[col] = df[col].map(to_float)

    trade_types = {"Buy", "Sell", "Assignment"}
    df["IsOption"] = df["Symbol"].map(parse_option_symbol)

    stk = df[
        df["Transaction Type"].isin(trade_types)
        & (~df["IsOption"])
        & df["Symbol"].notna()
        & (~df["Symbol"].str.strip().isin(["-", ""]))
        & (~df["Symbol"].str.contains(r"\.", regex=True))
        & df["Quantity"].notna()
        & df["Net Amount"].notna()
        & (df["Quantity"] != 0)
    ].copy()

    if stk.empty:
        print("Keine Aktien-Trades gefunden (Buy/Sell/Assignment, kein Optionssymbol).")
        return 0

    stk["Symbol"] = stk["Symbol"].str.strip().str.upper()
    stk = stk.sort_values("Date").reset_index(drop=True)
    stk["Seq"] = range(1, len(stk) + 1)  # stable ordering

    if args.date_to:
        asof = pd.to_datetime(args.date_to)
        stk = stk[stk["Date"] <= asof].copy()

    stk_with_pnl, open_lots = fifo_realized_stocks(stk)

    out = stk_with_pnl.copy()
    if args.year is not None:
        start = pd.Timestamp(datetime(args.year, 1, 1))
        end = pd.Timestamp(datetime(args.year, 12, 31, 23, 59, 59))
        out = out[(out["Date"] >= start) & (out["Date"] <= end)].copy()
    if args.date_from:
        out = out[out["Date"] >= pd.to_datetime(args.date_from)].copy()
    if args.date_to:
        out = out[out["Date"] <= pd.to_datetime(args.date_to)].copy()

    dmin = out["Date"].min()
    dmax = out["Date"].max()

    realized_total = float(out["RealizedStockEUR"].sum())

    rows = [
        ["Zeitraum (Auswertung)", f"{dmin.date()} … {dmax.date()}" if pd.notna(dmin) else "—"],
        ["Anzahl Aktien-Events (Buy/Sell/Assignment)", f"{int(len(out))}"],
        ["Realisierte Aktien P&L (FIFO, EUR)", eur_fmt(realized_total)],
    ]
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="github"))

    if args.by_symbol:
        g = out.groupby("Symbol", dropna=True)["RealizedStockEUR"].sum().sort_values(ascending=False)
        top = g.head(args.top).reset_index()
        top.columns = ["Symbol", "RealizedStockEUR"]
        top["RealizedStockEUR"] = top["RealizedStockEUR"].map(eur_fmt)
        print("\nTop Symbole (realisierte FIFO P&L):")
        print(tabulate(top, headers="keys", tablefmt="github", showindex=False))

    if args.show_open:
        if open_lots.empty:
            print("\nKeine offenen Aktien-Lots.")
        else:
            agg = open_lots.groupby("Symbol", as_index=False).agg(
                RestQty=("RestQty", "sum"),
                OpenCashflowEUR_Remaining=("OpenCashflowEUR_Remaining", "sum"),
            )
            agg = agg.sort_values("Symbol").reset_index(drop=True)

            print("\nOffene Aktien-Positionen (aggregiert):")
            tmp = agg.copy()
            tmp["OpenCashflowEUR_Remaining"] = tmp["OpenCashflowEUR_Remaining"].map(eur_fmt)
            print(tabulate(tmp.head(args.open_limit), headers="keys", tablefmt="github", showindex=False))

            if len(tmp) > args.open_limit:
                print(f"... ({len(tmp) - args.open_limit} weitere)")

    out_path = Path("stocks_diagramm.png")
    plot_monthly_realized_pnl(out, out_path=out_path)
    print(f"\nDiagramm gespeichert: {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())