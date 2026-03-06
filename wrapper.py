from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable

from matplotlib.ticker import FuncFormatter, MaxNLocator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

TRANSACTION_SECTION = "Transaction History"
SUMMARY_SECTION = "Summary"
TWOPLACES = Decimal("0.01")
ZERO = Decimal("0")

OCC_RE = re.compile(r"^(?P<root>[A-Z0-9]{1,6})(?P<yymmdd>\d{6})(?P<cp>[CP])(?P<strike8>\d{8})$")
EU_RE = re.compile(
    r"^(?P<cp>[PC])\s+(?P<und>[A-Z0-9.]+)\s+(?P<yyyymmdd>\d{8})\s+(?P<strike>\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
FX_PAIR_RE = re.compile(r"^[A-Z]{3}\.[A-Z]{3}$")
ACCRUAL_MONTH_RE = re.compile(r"\b(?:for|für)\s+([A-Za-zÄÖÜäöü]+)[-\s]?(\d{4})\b", re.IGNORECASE)
MONTHS = {
    "jan": 1, "januar": 1, "january": 1,
    "feb": 2, "februar": 2, "february": 2,
    "mär": 3, "maerz": 3, "märz": 3, "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "mai": 5, "may": 5,
    "jun": 6, "juni": 6, "june": 6,
    "jul": 7, "juli": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "okt": 10, "oct": 10, "oktober": 10, "october": 10,
    "nov": 11, "november": 11,
    "dez": 12, "dec": 12, "dezember": 12, "december": 12,
}


@dataclass(frozen=True)
class StatementInfo:
    base_currency: str | None


@dataclass
class Lot:
    open_dt: pd.Timestamp
    open_seq: int
    open_qty: Decimal
    rest_qty: Decimal
    remaining_open_cents: int


# ---------- generic helpers ----------

def expand_paths(single_path_or_glob: str) -> list[Path]:
    g = single_path_or_glob.strip()
    paths = sorted(Path().glob(g)) if any(ch in g for ch in "*?[") else [Path(g)]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit("Keine CSV gefunden.")
    return paths


def parse_decimal(x: object) -> Decimal | None:
    if x is None:
        return None
    s = str(x).strip()
    if s in {"", "-", "nan", "NaN", "None"}:
        return None
    s = s.replace(" ", "")
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def money_to_cents(v: Decimal | None) -> int:
    if v is None:
        return 0
    return int((v.quantize(TWOPLACES, rounding=ROUND_HALF_UP) * 100).to_integral_value())


def cents_to_dec(cents: int) -> Decimal:
    return Decimal(cents) / Decimal(100)


def dec_fmt(x: Decimal | int | float) -> str:
    d = x if isinstance(x, Decimal) else Decimal(str(x))
    s = f"{d.quantize(TWOPLACES, rounding=ROUND_HALF_UP):,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct_fmt(x: Decimal | None) -> str:
    if x is None:
        return "—"
    return f"{x.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)} %".replace(".", ",")


def parse_option_symbol(symbol: object) -> bool:
    s = str(symbol or "").strip()
    if s in {"", "-", "nan"}:
        return False
    key = re.sub(r"\s+", "", s).upper()
    return bool(OCC_RE.match(key) or EU_RE.match(s.upper()))


def is_fx_pair(symbol: object) -> bool:
    return bool(FX_PAIR_RE.match(str(symbol or "").strip().upper()))


def extract_accrual_month(description: object) -> pd.Timestamp | pd.NaT:
    m = ACCRUAL_MONTH_RE.search(str(description or ""))
    if not m:
        return pd.NaT
    month_key = m.group(1).strip().lower()
    month = MONTHS.get(month_key)
    if month is None:
        return pd.NaT
    return pd.Timestamp(year=int(m.group(2)), month=month, day=1)


def sign_decimal(x: Decimal) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


# ---------- reading ----------

def read_sections(paths: Iterable[Path]) -> tuple[pd.DataFrame, StatementInfo]:
    frames: list[pd.DataFrame] = []
    base_currency: str | None = None

    for p in paths:
        tx_header: list[str] | None = None
        tx_rows: list[list[str]] = []

        with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            for r in reader:
                if not r or len(r) < 3:
                    continue
                section = (r[0] or "").strip()
                kind = (r[1] or "").strip()

                if section == SUMMARY_SECTION and kind == "Data" and len(r) >= 4:
                    field_name = (r[2] or "").strip()
                    field_value = (r[3] or "").strip()
                    if field_name in {"Basiswährung", "Base Currency"} and field_value:
                        if base_currency is None:
                            base_currency = field_value
                        elif base_currency != field_value:
                            raise SystemExit(
                                f"Mischung mehrerer Basiswährungen gefunden: {base_currency} und {field_value}."
                            )

                if section != TRANSACTION_SECTION:
                    continue
                if kind == "Header":
                    tx_header = [c.strip() for c in r[2:]]
                elif kind == "Data":
                    if tx_header is None:
                        raise ValueError(f"Header fehlt vor Daten in {p}")
                    tx_rows.append(r[2:])

        if tx_header is None:
            raise ValueError(f"Keine '{TRANSACTION_SECTION}' Sektion in {p}")

        df = pd.DataFrame(tx_rows, columns=tx_header)
        df["SourceFile"] = str(p)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out.columns = [c.strip() for c in out.columns]
    out["_RowNo"] = np.arange(1, len(out) + 1)
    return out, StatementInfo(base_currency=base_currency)


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    if out["Date"].isna().any():
        raise SystemExit(f"{int(out['Date'].isna().sum())} Datumswerte konnten nicht geparst werden.")
    for col in ["Transaction Type", "Symbol", "Description", "Price Currency", "Account"]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
    for col in ["Quantity", "Price", "Gross Amount", "Commission", "Net Amount"]:
        if col in out.columns:
            out[col + "Dec"] = out[col].map(parse_decimal)
    out["IsOption"] = out["Symbol"].map(parse_option_symbol)
    out["IsFxPair"] = out["Symbol"].map(is_fx_pair)
    out["Month"] = out["Date"].dt.to_period("M").dt.to_timestamp()
    return out


# ---------- capital basis ----------

def build_monthly_contributed_capital(df: pd.DataFrame, starting_capital: Decimal) -> pd.DataFrame:
    all_months = pd.Series(pd.to_datetime(df["Month"].dropna().unique())).sort_values().tolist()
    if not all_months:
        return pd.DataFrame(columns=["Month", "CapitalBasisCents"])

    ext = df[df["Transaction Type"].isin(["Deposit", "Withdraw"])].copy()
    ext["ExternalCashCents"] = ext["Net AmountDec"].map(money_to_cents)
    flow_map: dict[pd.Timestamp, int] = {}
    if not ext.empty:
        monthly_flows = ext.groupby("Month", as_index=False)["ExternalCashCents"].sum().sort_values("Month")
        flow_map = {pd.Timestamp(m): int(v) for m, v in zip(monthly_flows["Month"], monthly_flows["ExternalCashCents"])}

    running_cents = int((starting_capital * 100).to_integral_value(rounding=ROUND_HALF_UP))
    rows: list[dict[str, object]] = []
    for month in all_months:
        ts_month = pd.Timestamp(month)
        running_cents += flow_map.get(ts_month, 0)
        rows.append({
            "Month": ts_month,
            "CapitalBasisCents": running_cents,
        })

    return pd.DataFrame(rows)


# ---------- options ----------

def summarize_options(df: pd.DataFrame, base_currency: str, capital_monthly: pd.DataFrame) -> tuple[list[list[str]], pd.DataFrame]:
    opt = df[df["Transaction Type"].isin(["Buy", "Sell"]) & df["IsOption"]].copy()
    if opt.empty:
        monthly = pd.DataFrame(columns=[
            "Month", "MonthLabel", "NetReceivedCents", "NetPaidCents", "NetCashflowCents",
            "GrossReceivedCents", "GrossPaidCents", "CommSumCents", "CapitalBasisCents", "OptionReturnPct"
        ])
        rows = [["Zeitraum", "—"], ["Basiswährung", base_currency], ["Option-Zeilen (Buy/Sell)", "0"]]
        return rows, monthly

    opt["NetCents"] = opt["Net AmountDec"].map(money_to_cents)
    opt["GrossCents"] = opt["Gross AmountDec"].map(money_to_cents)
    opt["CommCents"] = opt["CommissionDec"].map(money_to_cents)
    opt["ContractsAbs"] = opt["QuantityDec"].map(lambda d: abs(int(d)) if d is not None else 0)

    sell_rows = int((opt["Transaction Type"] == "Sell").sum())
    buy_rows = int((opt["Transaction Type"] == "Buy").sum())
    sell_contracts = int(opt.loc[opt["Transaction Type"] == "Sell", "ContractsAbs"].sum())
    buy_contracts = int(opt.loc[opt["Transaction Type"] == "Buy", "ContractsAbs"].sum())

    net_received_c = int(opt.loc[opt["NetCents"] > 0, "NetCents"].sum())
    net_paid_c = int((-opt.loc[opt["NetCents"] < 0, "NetCents"].sum()))
    net_total_c = int(opt["NetCents"].sum())
    gross_received_c = int(opt.loc[opt["GrossCents"] > 0, "GrossCents"].sum())
    gross_paid_c = int((-opt.loc[opt["GrossCents"] < 0, "GrossCents"].sum()))
    commission_total_c = int(opt["CommCents"].sum())

    dmin = opt["Date"].min()
    dmax = opt["Date"].max()
    period = f"{dmin.date()} … {dmax.date()}" if pd.notna(dmin) and pd.notna(dmax) else "—"

    monthly = opt.groupby("Month", as_index=False).agg(
        NetCashflowCents=("NetCents", "sum"),
        NetReceivedCents=("NetCents", lambda s: int(s[s > 0].sum())),
        NetPaidCents=("NetCents", lambda s: int((-s[s < 0].sum()))),
        GrossReceivedCents=("GrossCents", lambda s: int(s[s > 0].sum())),
        GrossPaidCents=("GrossCents", lambda s: int((-s[s < 0].sum()))),
        CommSumCents=("CommCents", "sum"),
    ).sort_values("Month").reset_index(drop=True)

    monthly = monthly.merge(capital_monthly, on="Month", how="left")
    monthly[["CapitalBasisCents"]] = monthly[["CapitalBasisCents"]].fillna(0).astype(int)
    monthly["OptionReturnPct"] = monthly.apply(
        lambda r: (Decimal(int(r["NetCashflowCents"])) / Decimal(int(r["CapitalBasisCents"])) * Decimal(100))
        if int(r["CapitalBasisCents"]) != 0 else None,
        axis=1,
    )
    monthly["MonthLabel"] = monthly["Month"].dt.strftime("%Y-%m")

    final_capital_c = int(capital_monthly["CapitalBasisCents"].iloc[-1]) if not capital_monthly.empty else 0
    total_return_pct = (Decimal(net_total_c) / Decimal(final_capital_c) * Decimal(100)) if final_capital_c else None

    rows = [
        ["Zeitraum", period],
        ["Basiswährung", base_currency],
        ["Option-Zeilen (Buy/Sell)", str(len(opt))],
        ["Sell-Zeilen", str(sell_rows)],
        ["Buy-Zeilen", str(buy_rows)],
        ["Verkaufte Kontrakte", str(sell_contracts)],
        ["Gekaufte Kontrakte", str(buy_contracts)],
        [f"Netto erhalten ({base_currency})", dec_fmt(cents_to_dec(net_received_c))],
        [f"Netto gezahlt ({base_currency})", dec_fmt(cents_to_dec(net_paid_c))],
        [f"Netto Summe ({base_currency})", dec_fmt(cents_to_dec(net_total_c))],
        [f"Brutto erhalten ({base_currency})", dec_fmt(cents_to_dec(gross_received_c))],
        [f"Brutto gezahlt ({base_currency})", dec_fmt(cents_to_dec(gross_paid_c))],
        [f"Provision Summe ({base_currency}, signed)", dec_fmt(cents_to_dec(commission_total_c))],
        [f"Kapitalbasis per Periodenende ({base_currency})", dec_fmt(cents_to_dec(final_capital_c))],
        ["Gesamt-Optionsperformance auf Kapitalbasis", pct_fmt(total_return_pct)],
    ]
    return rows, monthly


# ---------- stocks ----------

def fifo_realized_stocks(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df = trades.copy().sort_values(["Date", "Seq"], kind="mergesort").reset_index(drop=True)
    df["RealizedStockDec"] = Decimal("0")

    lots: dict[str, list[Lot]] = {}
    warnings: list[str] = []
    warned_symbols: set[str] = set()

    for idx, r in df.iterrows():
        sym = str(r["Symbol"]).strip().upper()
        dt = r["Date"]
        seq = int(r["Seq"])
        qty = r["QuantityDec"]
        net_cents_total = money_to_cents(r["Net AmountDec"])
        if qty is None or qty == 0:
            continue

        if sym not in lots:
            lots[sym] = []
            if qty < 0 and sym not in warned_symbols:
                warnings.append(
                    f"{sym}: erster Event im Datensatz ist ein Sell/Abgang ({dt.date()}). Ohne Vorhistorie kann FIFO falsch sein."
                )
                warned_symbols.add(sym)

        q_remaining = qty
        trade_remaining_qty_abs = abs(qty)
        trade_remaining_cents = net_cents_total
        lot_list = lots[sym]
        realized_cents_this_trade = 0

        i = 0
        while i < len(lot_list) and sign_decimal(q_remaining) != 0:
            lot = lot_list[i]
            if sign_decimal(lot.rest_qty) == 0:
                i += 1
                continue
            if sign_decimal(lot.rest_qty) != -sign_decimal(q_remaining):
                i += 1
                continue

            match_qty = min(abs(q_remaining), abs(lot.rest_qty))
            if match_qty <= 0:
                i += 1
                continue

            lot_remaining_qty_abs = abs(lot.rest_qty)
            open_alloc = int(round(lot.remaining_open_cents * float(match_qty / lot_remaining_qty_abs)))
            close_alloc = int(round(trade_remaining_cents * float(match_qty / trade_remaining_qty_abs))) if trade_remaining_qty_abs > 0 else 0
            realized_cents_this_trade += open_alloc + close_alloc

            lot.rest_qty = lot.rest_qty + (-match_qty if lot.rest_qty > 0 else match_qty)
            lot.remaining_open_cents -= open_alloc
            q_remaining = q_remaining + (-match_qty if q_remaining > 0 else match_qty)
            trade_remaining_qty_abs -= match_qty
            trade_remaining_cents -= close_alloc

            if abs(lot.rest_qty) == 0:
                lot.remaining_open_cents = 0
            if abs(q_remaining) == 0:
                q_remaining = Decimal("0")
            i += 1

        if sign_decimal(q_remaining) != 0:
            lot_list.append(
                Lot(
                    open_dt=dt,
                    open_seq=seq,
                    open_qty=q_remaining,
                    rest_qty=q_remaining,
                    remaining_open_cents=trade_remaining_cents,
                )
            )

        df.at[idx, "RealizedStockDec"] = Decimal(realized_cents_this_trade) / Decimal(100)

    lot_rows = []
    for sym, lot_list in lots.items():
        for lot in lot_list:
            if abs(lot.rest_qty) == 0:
                continue
            lot_rows.append(
                {
                    "Symbol": sym,
                    "OpenDate": lot.open_dt,
                    "OpenSeq": lot.open_seq,
                    "OpenQty": float(lot.open_qty),
                    "RestQty": float(lot.rest_qty),
                    "OpenCashflowRemaining": Decimal(lot.remaining_open_cents) / Decimal(100),
                }
            )
    lots_df = pd.DataFrame(lot_rows)
    if not lots_df.empty:
        lots_df = lots_df.sort_values(["Symbol", "OpenDate", "OpenSeq"]).reset_index(drop=True)
    return df, lots_df, warnings


def summarize_stocks(df: pd.DataFrame, base_currency: str, capital_monthly: pd.DataFrame) -> tuple[list[list[str]], pd.DataFrame, pd.DataFrame, list[str]]:
    trade_types = {"Buy", "Sell", "Assignment"}
    stk = df[
        df["Transaction Type"].isin(trade_types)
        & (~df["IsOption"])
        & (~df["IsFxPair"])
        & (~df["Symbol"].str.strip().isin(["", "-", "nan"]))
        & df["QuantityDec"].notna()
        & df["Net AmountDec"].notna()
        & (df["QuantityDec"] != 0)
    ].copy()

    if stk.empty:
        rows = [["Zeitraum", "—"], ["Basiswährung", base_currency], ["Aktien-Events", "0"]]
        monthly = pd.DataFrame(columns=[
            "Month", "MonthLabel", "EventCount", "NetReceivedCents", "NetPaidCents",
            "RealizedPnLCents", "CommSumCents", "CapitalBasisCents", "StockReturnPct"
        ])
        return rows, monthly, pd.DataFrame(), []

    stk = stk.sort_values(["Date", "_RowNo"], kind="mergesort").reset_index(drop=True)
    stk["Seq"] = np.arange(1, len(stk) + 1)
    stk["NetCents"] = stk["Net AmountDec"].map(money_to_cents)
    stk["CommCents"] = stk["CommissionDec"].map(money_to_cents)
    stk["SharesAbs"] = stk["QuantityDec"].map(lambda d: abs(int(d)) if d is not None else 0)

    stk_with_pnl, open_lots, warnings = fifo_realized_stocks(stk)

    dmin = stk_with_pnl["Date"].min()
    dmax = stk_with_pnl["Date"].max()
    period = f"{dmin.date()} … {dmax.date()}" if pd.notna(dmin) and pd.notna(dmax) else "—"

    event_count = int(len(stk_with_pnl))
    buy_rows = int((stk_with_pnl["Transaction Type"] == "Buy").sum())
    sell_rows = int((stk_with_pnl["Transaction Type"] == "Sell").sum())
    assignment_rows = int((stk_with_pnl["Transaction Type"] == "Assignment").sum())
    bought_shares = int(stk_with_pnl.loc[stk_with_pnl["QuantityDec"] > 0, "SharesAbs"].sum())
    sold_shares = int(stk_with_pnl.loc[stk_with_pnl["QuantityDec"] < 0, "SharesAbs"].sum())
    realized_total = sum(stk_with_pnl["RealizedStockDec"], Decimal("0"))
    net_received_c = int(stk_with_pnl.loc[stk_with_pnl["NetCents"] > 0, "NetCents"].sum())
    net_paid_c = int((-stk_with_pnl.loc[stk_with_pnl["NetCents"] < 0, "NetCents"].sum()))
    commission_total_c = int(stk_with_pnl["CommCents"].sum())

    open_position_count = int(open_lots["Symbol"].nunique()) if not open_lots.empty else 0
    open_lot_count = int(len(open_lots)) if not open_lots.empty else 0
    open_shares_total = int(round(open_lots["RestQty"].abs().sum())) if not open_lots.empty else 0

    monthly = stk_with_pnl.groupby("Month", as_index=False).agg(
        EventCount=("Symbol", "size"),
        NetReceivedCents=("NetCents", lambda s: int(s[s > 0].sum())),
        NetPaidCents=("NetCents", lambda s: int((-s[s < 0].sum()))),
        RealizedPnLCents=("RealizedStockDec", lambda s: int((sum(s, Decimal('0')) * Decimal(100)).to_integral_value(rounding=ROUND_HALF_UP))),
        CommSumCents=("CommCents", "sum"),
    ).sort_values("Month").reset_index(drop=True)
    monthly = monthly.merge(capital_monthly, on="Month", how="left")
    monthly[["CapitalBasisCents"]] = monthly[["CapitalBasisCents"]].fillna(0).astype(int)
    monthly["StockReturnPct"] = monthly.apply(
        lambda r: (Decimal(int(r["RealizedPnLCents"])) / Decimal(int(r["CapitalBasisCents"])) * Decimal(100))
        if int(r["CapitalBasisCents"]) != 0 else None,
        axis=1,
    )
    monthly["MonthLabel"] = monthly["Month"].dt.strftime("%Y-%m")

    final_capital_c = int(capital_monthly["CapitalBasisCents"].iloc[-1]) if not capital_monthly.empty else 0
    total_return_pct = (realized_total / cents_to_dec(final_capital_c) * Decimal(100)) if final_capital_c else None

    rows = [
        ["Zeitraum", period],
        ["Basiswährung", base_currency],
        ["Aktien-Events (Buy/Sell/Assignment)", str(event_count)],
        ["Buy-Zeilen", str(buy_rows)],
        ["Sell-Zeilen", str(sell_rows)],
        ["Assignment-Zeilen", str(assignment_rows)],
        ["Gekaufte Aktien", str(bought_shares)],
        ["Verkaufte Aktien", str(sold_shares)],
        [f"Netto erhalten ({base_currency})", dec_fmt(cents_to_dec(net_received_c))],
        [f"Netto gezahlt ({base_currency})", dec_fmt(cents_to_dec(net_paid_c))],
        [f"Provision Summe ({base_currency}, signed)", dec_fmt(cents_to_dec(commission_total_c))],
        [f"Realisierte FIFO P&L ({base_currency}, stock-only)", dec_fmt(realized_total)],
        [f"Kapitalbasis per Periodenende ({base_currency})", dec_fmt(cents_to_dec(final_capital_c))],
        ["Gesamt-Aktienperformance auf Kapitalbasis", pct_fmt(total_return_pct)],
        ["Offene Positionen", str(open_position_count)],
        ["Offene Lots", str(open_lot_count)],
        ["Offene Aktien gesamt", str(open_shares_total)],
    ]
    return rows, monthly, open_lots, warnings


# ---------- cash ----------

def summarize_cash(df: pd.DataFrame, base_currency: str) -> tuple[list[list[str]], pd.DataFrame]:
    deposits = df[df["Transaction Type"].eq("Deposit")].copy()
    withdrawals = df[df["Transaction Type"].eq("Withdraw")].copy()
    fx = df[df["Transaction Type"].eq("Forex Trade Component") | df["IsFxPair"]].copy()
    base_pair = f"{base_currency}.USD"
    base_pair_rows = fx[fx["Symbol"].str.upper().eq(base_pair)].copy() if not fx.empty else fx.copy()

    dep_total = cents_to_dec(int(deposits["Net AmountDec"].map(money_to_cents).sum()))
    wdr_total = cents_to_dec(int((-withdrawals.loc[withdrawals["Net AmountDec"].map(lambda x: x is not None and x < 0), "Net AmountDec"].map(money_to_cents).sum())))
    net_cash_in = dep_total - wdr_total

    fx_comm_signed = sum((v for v in fx["CommissionDec"] if v is not None), ZERO)
    fx_base_signed = sum((v for v in fx["Net AmountDec"] if v is not None), ZERO)
    fx_comm_paid = sum(((-v) for v in fx["CommissionDec"] if v is not None and v < 0), ZERO)
    fx_base_paid = sum(((-v) for v in fx["Net AmountDec"] if v is not None and v < 0), ZERO)
    fx_total_paid = fx_comm_paid + fx_base_paid
    base_to_quote = sum(((-v) for v in base_pair_rows["QuantityDec"] if v is not None and v < 0), ZERO)
    quote_to_base = sum((v for v in base_pair_rows["QuantityDec"] if v is not None and v > 0), ZERO)
    net_converted = base_to_quote - quote_to_base

    dmin = df["Date"].min()
    dmax = df["Date"].max()
    period = f"{dmin.date()} … {dmax.date()}" if pd.notna(dmin) and pd.notna(dmax) else "—"

    rows = [
        ["Zeitraum", period],
        ["Basiswährung", base_currency],
        [f"Einzahlungen gesamt ({base_currency})", dec_fmt(dep_total)],
        [f"Auszahlungen gesamt ({base_currency})", dec_fmt(wdr_total)],
        [f"Netto Cash In ({base_currency})", dec_fmt(net_cash_in)],
        [f"{base_currency} → USD umgerechnet ({base_currency})", dec_fmt(base_to_quote)],
        [f"USD → {base_currency} umgerechnet ({base_currency})", dec_fmt(quote_to_base)],
        [f"Netto in USD gewechselt ({base_currency})", dec_fmt(net_converted)],
        [f"FX Provision Summe ({base_currency}, signed)", dec_fmt(fx_comm_signed)],
        [f"FX Basisbetrags-Komponente Summe ({base_currency}, signed)", dec_fmt(fx_base_signed)],
        [f"FX Total Paid ({base_currency})", dec_fmt(fx_total_paid)],
    ]

    monthly_map: dict[pd.Timestamp, dict[str, Decimal]] = defaultdict(lambda: {
        "Deposits": ZERO,
        "Withdrawals": ZERO,
        "BaseToQuoteVolume": ZERO,
        "FxFeesPaid": ZERO,
        "FxBaseAmountPaid": ZERO,
    })
    for _, r in deposits.iterrows():
        monthly_map[r["Month"]]["Deposits"] += r["Net AmountDec"] or ZERO
    for _, r in withdrawals.iterrows():
        if r["Net AmountDec"] is not None and r["Net AmountDec"] < 0:
            monthly_map[r["Month"]]["Withdrawals"] += -r["Net AmountDec"]
    for _, r in base_pair_rows.iterrows():
        if r["QuantityDec"] is not None and r["QuantityDec"] < 0:
            monthly_map[r["Month"]]["BaseToQuoteVolume"] += -r["QuantityDec"]
    for _, r in fx.iterrows():
        if r["CommissionDec"] is not None and r["CommissionDec"] < 0:
            monthly_map[r["Month"]]["FxFeesPaid"] += -r["CommissionDec"]
        if r["Net AmountDec"] is not None and r["Net AmountDec"] < 0:
            monthly_map[r["Month"]]["FxBaseAmountPaid"] += -r["Net AmountDec"]

    monthly = pd.DataFrame([
        {
            "Month": m,
            "Deposits": float(v["Deposits"]),
            "Withdrawals": float(v["Withdrawals"]),
            "BaseToQuoteVolume": float(v["BaseToQuoteVolume"]),
            "FxFeesPaid": float(v["FxFeesPaid"]),
            "FxBaseAmountPaid": float(v["FxBaseAmountPaid"]),
        }
        for m, v in sorted(monthly_map.items())
    ])
    if monthly.empty:
        monthly = pd.DataFrame(columns=["Month", "MonthLabel", "Deposits", "Withdrawals", "BaseToQuoteVolume", "FxFeesPaid", "FxBaseAmountPaid"])
    else:
        monthly["MonthLabel"] = monthly["Month"].dt.strftime("%Y-%m")
    return rows, monthly


# ---------- interest ----------

def summarize_interest(df: pd.DataFrame, base_currency: str, group_by: str) -> tuple[list[list[str]], pd.DataFrame]:
    strict = df["Transaction Type"].isin(["Debit Interest", "Credit Interest"])
    desc = df.get("Description", pd.Series(index=df.index, dtype=str)).astype(str)
    hinted = desc.str.contains(r"sollzinsen|habenzinsen|interest", case=False, na=False)
    interest = df[strict | hinted].copy()
    if interest.empty:
        rows = [["Zeitraum", "—"], ["Basiswährung", base_currency], ["Zins-Zeilen", "0"]]
        monthly = pd.DataFrame(columns=["Month", "MonthLabel", "InterestNet"])
        return rows, monthly

    interest["Detection"] = np.where(strict.loc[interest.index], "type", "description")
    interest["NetRounded"] = interest["Net AmountDec"].map(lambda x: (x or ZERO).quantize(TWOPLACES, rounding=ROUND_HALF_UP))

    total_net = sum(interest["NetRounded"], Decimal("0.00"))
    total_paid = -sum((v for v in interest["NetRounded"] if v < 0), Decimal("0.00"))
    total_received = sum((v for v in interest["NetRounded"] if v > 0), Decimal("0.00"))
    dmin = interest["Date"].min()
    dmax = interest["Date"].max()
    period = f"{dmin.date()} … {dmax.date()}" if pd.notna(dmin) and pd.notna(dmax) else "—"

    if group_by == "posting":
        interest["MonthGroup"] = interest["Month"]
    else:
        interest["MonthGroup"] = interest["Description"].map(extract_accrual_month)
        interest["MonthGroup"] = interest["MonthGroup"].fillna(interest["Month"])

    monthly = (
        interest.groupby("MonthGroup", as_index=False)["NetRounded"]
        .agg(lambda s: sum(s, Decimal("0.00")))
        .sort_values("MonthGroup")
        .reset_index(drop=True)
    )
    monthly["Month"] = monthly["MonthGroup"]
    monthly["MonthLabel"] = monthly["Month"].dt.strftime("%Y-%m")
    monthly["InterestNet"] = monthly["NetRounded"].map(float)

    detection_counts = interest["Detection"].value_counts().to_dict()
    rows = [
        ["Zeitraum", period],
        ["Basiswährung", base_currency],
        [f"Netto Zinsen gesamt ({base_currency}, signed)", dec_fmt(total_net)],
        [f"Davon gezahlt ({base_currency})", dec_fmt(total_paid)],
        [f"Davon erhalten ({base_currency})", dec_fmt(total_received)],
        ["Erkennung via Transaction Type", str(detection_counts.get("type", 0))],
        ["Erkennung via Description-Hinweis", str(detection_counts.get("description", 0))],
        ["Monatslogik", "Buchungsmonat" if group_by == "posting" else "Zinsmonat aus Description / Fallback Buchungsmonat"],
    ]
    return rows, monthly[["Month", "MonthLabel", "InterestNet"]]


# ---------- charts ----------

def _chart_figsize(n_months: int, n_series: int) -> tuple[float, float]:
    width = max(13.0, 5.0 + n_months * (0.95 + 0.08 * n_series))
    height = 7.4 if n_series <= 3 else 7.8
    return width, height


def _label_fontsize(n_months: int, max_abs_value: float, grouped: bool) -> float:
    fs = 8.0
    if n_months >= 8:
        fs = 7.4
    if n_months >= 10:
        fs = 6.9
    if n_months >= 12:
        fs = 6.4
    if max_abs_value >= 100_000:
        fs = min(fs, 6.4)
    if grouped and n_months >= 10:
        fs = min(fs, 6.2)
    return fs


def _label_rotation(n_months: int, grouped: bool) -> int:
    if grouped and n_months >= 10:
        return 90
    return 0


def _safe_ylim(values: list[np.ndarray], pad_ratio: float = 0.24) -> tuple[float, float]:
    arrays = [np.asarray(v, dtype=float) for v in values if len(v)]
    if not arrays:
        return -10.0, 10.0

    flat = np.concatenate(arrays)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return -10.0, 10.0

    y_min = float(min(0.0, flat.min()))
    y_max = float(max(0.0, flat.max()))
    max_abs = max(abs(y_min), abs(y_max), 1.0)

    # Mehr Headroom für große Beträge, damit Labels nie oben abgeschnitten werden
    if max_abs >= 100_000:
        pad = max(max_abs * 0.34, 5000.0)
    elif max_abs >= 10_000:
        pad = max(max_abs * 0.28, 800.0)
    else:
        pad = max(max_abs * pad_ratio, 20.0)

    if y_min >= 0:
        return -pad * 0.08, y_max + pad
    if y_max <= 0:
        return y_min - pad, pad * 0.08
    return y_min - pad, y_max + pad


def _apply_axis_style(ax, title: str, ylabel: str, months: list[str]) -> None:
    ax.axhline(0, linewidth=1.15, color="#64748B", alpha=0.9)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Monat", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    ax.set_xticks(np.arange(len(months)))
    ax.set_xticklabels(months, rotation=45, ha="right", fontsize=9)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: dec_fmt(v)))

    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.28)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94A3B8")
    ax.spines["bottom"].set_color("#94A3B8")

    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", pad=6)


def add_value_labels(
    ax,
    bars,
    *,
    fmt=dec_fmt,
    n_months: int = 12,
    grouped: bool = True,
) -> None:
    heights = [abs(float(b.get_height())) for b in bars if abs(float(b.get_height())) > 1e-12]
    max_abs_value = max(heights) if heights else 0.0
    fontsize = _label_fontsize(n_months=n_months, max_abs_value=max_abs_value, grouped=grouped)
    rotation = _label_rotation(n_months=n_months, grouped=grouped)

    y0, y1 = ax.get_ylim()
    span = max(abs(y1 - y0), 1.0)
    offset = span * 0.018
    min_offset = max(span * 0.012, 6.0 if max_abs_value >= 100_000 else 2.0)
    offset = max(offset, min_offset)

    for b in bars:
        h = float(b.get_height())
        if abs(h) < 1e-12:
            continue

        x = b.get_x() + b.get_width() / 2.0
        y = h + offset if h >= 0 else h - offset
        va = "bottom" if h >= 0 else "top"

        ax.text(
            x,
            y,
            fmt(h),
            ha="center",
            va=va,
            fontsize=fontsize,
            rotation=rotation,
            clip_on=False,
            zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.16",
                facecolor="white",
                edgecolor="none",
                alpha=0.82,
            ),
        )


def _finalize_chart(fig, ax, out_path: Path) -> None:
    fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.25)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def plot_options(monthly: pd.DataFrame, base_currency: str, out_path: Path) -> None:
    months = monthly["MonthLabel"].tolist() if not monthly.empty else []
    fig, ax = plt.subplots(figsize=_chart_figsize(len(months), 4))

    x = np.arange(len(months))
    w = 0.18

    cashflow = monthly["NetCashflowCents"].to_numpy(dtype=float) / 100.0 if not monthly.empty else np.array([])
    received = monthly["NetReceivedCents"].to_numpy(dtype=float) / 100.0 if not monthly.empty else np.array([])
    paid = monthly["NetPaidCents"].to_numpy(dtype=float) / 100.0 if not monthly.empty else np.array([])
    commission = -monthly["CommSumCents"].to_numpy(dtype=float) / 100.0 if not monthly.empty else np.array([])

    cashflow_colors = ["#16A34A" if v >= 0 else "#DC2626" for v in cashflow]

    b1 = ax.bar(x - 1.5 * w, cashflow, width=w, label="Netto Cashflow", color=cashflow_colors)
    b2 = ax.bar(x - 0.5 * w, received, width=w, label="Netto erhalten", color="#2563EB")
    b3 = ax.bar(x + 0.5 * w, paid, width=w, label="Netto gezahlt", color="#F59E0B")
    b4 = ax.bar(x + 1.5 * w, commission, width=w, label="Provision bezahlt", color="#111827")

    _apply_axis_style(ax, f"Optionen: Monatsübersicht ({base_currency})", base_currency, months)
    ylo, yhi = _safe_ylim([cashflow, received, -paid, -commission], pad_ratio=0.26)
    ax.set_ylim(ylo, yhi)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncols=2, fontsize=8, frameon=False)

    for bars in (b1, b2, b3, b4):
        add_value_labels(ax, bars, n_months=len(months), grouped=True)

    _finalize_chart(fig, ax, out_path)


def plot_stocks(monthly: pd.DataFrame, base_currency: str, out_path: Path) -> None:
    months = monthly["MonthLabel"].tolist() if not monthly.empty else []
    fig, ax = plt.subplots(figsize=_chart_figsize(len(months), 3))

    x = np.arange(len(months))
    w = 0.24

    realized = monthly["RealizedPnLCents"].to_numpy(dtype=float) / 100.0 if not monthly.empty else np.array([])
    received = monthly["NetReceivedCents"].to_numpy(dtype=float) / 100.0 if not monthly.empty else np.array([])
    paid = monthly["NetPaidCents"].to_numpy(dtype=float) / 100.0 if not monthly.empty else np.array([])

    realized_colors = ["#16A34A" if v >= 0 else "#DC2626" for v in realized]

    b1 = ax.bar(x - w, realized, width=w, label="Realisierte FIFO P&L", color=realized_colors)
    b2 = ax.bar(x, received, width=w, label="Netto erhalten", color="#2563EB")
    b3 = ax.bar(x + w, paid, width=w, label="Netto gezahlt", color="#F59E0B")

    _apply_axis_style(ax, f"Aktien: Monatsübersicht ({base_currency})", base_currency, months)
    ylo, yhi = _safe_ylim([realized, received, -paid], pad_ratio=0.26)
    ax.set_ylim(ylo, yhi)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncols=3, fontsize=8, frameon=False)

    for bars in (b1, b2, b3):
        add_value_labels(ax, bars, n_months=len(months), grouped=True)

    _finalize_chart(fig, ax, out_path)


def plot_cash(monthly: pd.DataFrame, base_currency: str, out_path: Path) -> None:
    months = monthly["MonthLabel"].tolist() if not monthly.empty else []
    fig, ax = plt.subplots(figsize=_chart_figsize(len(months), 5))

    x = np.arange(len(months))
    w = 0.17

    dep = monthly["Deposits"].to_numpy(dtype=float) if not monthly.empty else np.array([])
    wdr = monthly["Withdrawals"].to_numpy(dtype=float) if not monthly.empty else np.array([])
    conv = monthly["BaseToQuoteVolume"].to_numpy(dtype=float) if not monthly.empty else np.array([])
    fee = monthly["FxFeesPaid"].to_numpy(dtype=float) if not monthly.empty else np.array([])
    base_paid = monthly["FxBaseAmountPaid"].to_numpy(dtype=float) if not monthly.empty else np.array([])

    b1 = ax.bar(x - 2 * w, dep, width=w, label="Einzahlungen", color="#16A34A")
    b2 = ax.bar(x - 1 * w, wdr, width=w, label="Auszahlungen", color="#DC2626")
    b3 = ax.bar(x + 0 * w, conv, width=w, label=f"{base_currency} → USD", color="#2563EB")
    b4 = ax.bar(x + 1 * w, fee, width=w, label="FX Provision", color="#111827")
    b5 = ax.bar(x + 2 * w, base_paid, width=w, label="FX Basis-Komponente", color="#6B7280")

    _apply_axis_style(ax, f"Cash & FX: Monatsübersicht ({base_currency})", base_currency, months)
    ylo, yhi = _safe_ylim([dep, wdr, conv, fee, base_paid], pad_ratio=0.28)
    ax.set_ylim(ylo, yhi)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.03), ncols=3, fontsize=8, frameon=False)

    for bars in (b1, b2, b3, b4, b5):
        add_value_labels(ax, bars, n_months=len(months), grouped=True)

    _finalize_chart(fig, ax, out_path)


def plot_interest(monthly: pd.DataFrame, base_currency: str, out_path: Path) -> None:
    months = monthly["MonthLabel"].tolist() if not monthly.empty else []
    fig, ax = plt.subplots(figsize=_chart_figsize(len(months), 1))

    x = np.arange(len(months))
    y = monthly["InterestNet"].to_numpy(dtype=float) if not monthly.empty else np.array([])
    colors_interest = ["#16A34A" if v >= 0 else "#DC2626" for v in y]

    bars = ax.bar(x, y, color=colors_interest, width=0.55)

    _apply_axis_style(ax, f"Zinsen: Monatsübersicht ({base_currency})", base_currency, months)
    ylo, yhi = _safe_ylim([y], pad_ratio=0.30)
    ax.set_ylim(ylo, yhi)

    add_value_labels(ax, bars, n_months=len(months), grouped=False)

    _finalize_chart(fig, ax, out_path)

# ---------- pdf ----------

def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Heading1"], fontSize=18, leading=22, textColor=colors.HexColor("#0F172A"), spaceAfter=8))
    styles.add(ParagraphStyle(name="SectionNote", parent=styles["BodyText"], fontSize=8.5, leading=10.5, textColor=colors.HexColor("#475569"), spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.2, leading=10))
    return styles


def make_table(data: list[list[str]], col_widths: list[float] | None = None, header: bool = True) -> Table:
    table = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    table.setStyle(TableStyle(style))
    return table


def summary_table_from_rows(rows: list[list[str]]) -> Table:
    data = [["Kennzahl", "Wert"]] + rows
    return make_table(data, col_widths=[92 * mm, 78 * mm])


def monthly_table_options(monthly: pd.DataFrame, base_currency: str) -> Table:
    data = [["Monat", "Netto erhalten", "Netto gezahlt", "Netto Summe", "Provision", "Kapitalbasis", "Options-%"]]
    for _, r in monthly.iterrows():
        data.append([
            r["MonthLabel"],
            dec_fmt(cents_to_dec(int(r["NetReceivedCents"]))),
            dec_fmt(cents_to_dec(int(r["NetPaidCents"]))),
            dec_fmt(cents_to_dec(int(r["NetCashflowCents"]))),
            dec_fmt(cents_to_dec(int(r["CommSumCents"]))),
            dec_fmt(cents_to_dec(int(r["CapitalBasisCents"]))),
            pct_fmt(r["OptionReturnPct"]),
        ])
    return make_table(data, col_widths=[23 * mm, 28 * mm, 26 * mm, 24 * mm, 20 * mm, 27 * mm, 18 * mm])


def monthly_table_stocks(monthly: pd.DataFrame) -> Table:
    data = [["Monat", "Events", "Netto erhalten", "Netto gezahlt", "FIFO P&L", "Provision", "Kapitalbasis", "Aktien-%"]]
    for _, r in monthly.iterrows():
        data.append([
            r["MonthLabel"],
            str(int(r["EventCount"])),
            dec_fmt(cents_to_dec(int(r["NetReceivedCents"]))),
            dec_fmt(cents_to_dec(int(r["NetPaidCents"]))),
            dec_fmt(cents_to_dec(int(r["RealizedPnLCents"]))),
            dec_fmt(cents_to_dec(int(r["CommSumCents"]))),
            dec_fmt(cents_to_dec(int(r["CapitalBasisCents"]))),
            pct_fmt(r["StockReturnPct"]),
        ])
    return make_table(data, col_widths=[19 * mm, 14 * mm, 25 * mm, 24 * mm, 21 * mm, 18 * mm, 23 * mm, 16 * mm])


def monthly_table_cash(monthly: pd.DataFrame) -> Table:
    data = [["Monat", "Einzahlungen", "Auszahlungen", "FX Basis→USD", "FX Provision", "FX Basis-Komp."]]
    for _, r in monthly.iterrows():
        data.append([
            r["MonthLabel"],
            dec_fmt(r["Deposits"]),
            dec_fmt(r["Withdrawals"]),
            dec_fmt(r["BaseToQuoteVolume"]),
            dec_fmt(r["FxFeesPaid"]),
            dec_fmt(r["FxBaseAmountPaid"]),
        ])
    return make_table(data, col_widths=[25 * mm, 29 * mm, 29 * mm, 30 * mm, 24 * mm, 31 * mm])


def monthly_table_interest(monthly: pd.DataFrame) -> Table:
    data = [["Monat", "Netto Zinsen"]]
    for _, r in monthly.iterrows():
        data.append([r["MonthLabel"], dec_fmt(r["InterestNet"])])
    return make_table(data, col_widths=[35 * mm, 40 * mm])


def page_header(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(18 * mm, height - 16 * mm, width - 18 * mm, height - 16 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#475569"))
    canvas.drawRightString(width - 18 * mm, 10 * mm, f"Seite {doc.page}")
    canvas.restoreState()


def build_pdf(
    out_pdf: Path,
    base_currency: str,
    options_rows: list[list[str]],
    options_monthly: pd.DataFrame,
    stocks_rows: list[list[str]],
    stocks_monthly: pd.DataFrame,
    open_lots: pd.DataFrame,
    stock_warnings: list[str],
    cash_rows: list[list[str]],
    cash_monthly: pd.DataFrame,
    interest_rows: list[list[str]],
    interest_monthly: pd.DataFrame,
    chart_paths: dict[str, Path],
    capital_note: str,
):
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=20 * mm,
        bottomMargin=14 * mm,
    )
    story = []

    def add_section(title: str, note: str, summary_rows: list[list[str]], monthly_tbl: Table, chart_key: str, extra: list | None = None, chart_height_mm: float = 58):
        story.append(Paragraph(title, styles["ReportTitle"]))
        story.append(Paragraph(note, styles["SectionNote"]))
        story.append(summary_table_from_rows(summary_rows))
        story.append(Spacer(1, 5 * mm))
        story.append(Paragraph("Monatsübersicht", styles["Heading3"]))
        story.append(monthly_tbl)
        story.append(Spacer(1, 4 * mm))
        pass
        if extra:
            story.extend(extra)
        story.append(PageBreak())

    add_section(
        "Optionen",
        f"Optionen AMK",
        options_rows,
        monthly_table_options(options_monthly, base_currency),
        "options",
        chart_height_mm=56,
    )

    add_section(
        "Aktien",
        f"Aktien AMK",
        stocks_rows,
        monthly_table_stocks(stocks_monthly),
        "stocks",
        None,
        chart_height_mm=50,
    )

    add_section(
        "Cash & FX",
        "Cash AMK",
        cash_rows,
        monthly_table_cash(cash_monthly),
        "cash",
        chart_height_mm=56,
    )

    story.append(Paragraph("Interest", styles["ReportTitle"]))
    story.append(Paragraph("Zinsen AMK", styles["SectionNote"]))
    story.append(summary_table_from_rows(interest_rows))
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("Monatsübersicht", styles["Heading3"]))
    story.append(monthly_table_interest(interest_monthly))
    story.append(Spacer(1, 4 * mm))
    pass

    doc.build(story, onFirstPage=page_header, onLaterPages=page_header)


# ---------- main ----------

def ensure_structure(base_dir: Path) -> dict[str, Path]:
    output_dir = base_dir / "output"
    reports_dir = output_dir / "reports"
    charts_dir = output_dir / "charts"
    logs_dir = output_dir / "logs"
    for p in [output_dir, reports_dir, charts_dir, logs_dir]:
        p.mkdir(parents=True, exist_ok=True)
    return {"output": output_dir, "reports": reports_dir, "charts": charts_dir, "logs": logs_dir}


def main() -> int:
    ap = argparse.ArgumentParser(description="Wrapper-Report für Optionen, Aktien, Cash/FX und Zinsen aus LYNX/IBKR Transaction-History CSV.")
    ap.add_argument("--csv", required=True, help="CSV Pfad oder Glob, z.B. data/U24066232.TRANSACTIONS.YTD.csv")
    ap.add_argument("--starting-capital", default="0", help="Opening Capital in Statement-Basiswährung vor dem ersten CSV-Datum, z.B. 50000")
    ap.add_argument("--interest-group-by", choices=["posting", "accrual"], default="posting")
    ap.add_argument("--output-dir", default="output", help="Output-Ordner für PDF und Charts")
    args = ap.parse_args()

    project_root = Path.cwd()
    dirs = ensure_structure(project_root)
    if args.output_dir != "output":
        custom = Path(args.output_dir)
        (custom / "reports").mkdir(parents=True, exist_ok=True)
        (custom / "charts").mkdir(parents=True, exist_ok=True)
        dirs["reports"] = custom / "reports"
        dirs["charts"] = custom / "charts"

    paths = expand_paths(args.csv)
    raw_df, statement = read_sections(paths)
    df = prepare_df(raw_df)
    base_currency = statement.base_currency or "EUR"

    starting_capital = parse_decimal(args.starting_capital)
    if starting_capital is None:
        raise SystemExit("--starting-capital ist nicht parsebar.")

    cap_monthly = build_monthly_contributed_capital(df, starting_capital)
    capital_note = (
        "Startwert vor erstem CSV-Datum wird via --starting-capital ergänzt."
        if starting_capital != 0
        else "Ohne Opening Capital ist die Kapitalbasis nur dann exakt, wenn der Export ab Kontostart beginnt."
    )

    options_rows, options_monthly = summarize_options(df, base_currency, cap_monthly)
    stocks_rows, stocks_monthly, open_lots, stock_warnings = summarize_stocks(df, base_currency, cap_monthly)
    cash_rows, cash_monthly = summarize_cash(df, base_currency)
    interest_rows, interest_monthly = summarize_interest(df, base_currency, args.interest_group_by)

    chart_paths = {
        "options": dirs["charts"] / "options_monthly.png",
        "stocks": dirs["charts"] / "stocks_monthly.png",
        "cash": dirs["charts"] / "cash_monthly.png",
        "interest": dirs["charts"] / f"interest_monthly_{args.interest_group_by}.png",
    }

    plot_options(options_monthly, base_currency, chart_paths["options"])
    plot_stocks(stocks_monthly, base_currency, chart_paths["stocks"])
    plot_cash(cash_monthly, base_currency, chart_paths["cash"])
    plot_interest(interest_monthly, base_currency, chart_paths["interest"])

    pdf_path = dirs["reports"] / "gesamtuebersicht_report.pdf"
    build_pdf(
        pdf_path,
        base_currency,
        options_rows,
        options_monthly,
        stocks_rows,
        stocks_monthly,
        open_lots,
        stock_warnings,
        cash_rows,
        cash_monthly,
        interest_rows,
        interest_monthly,
        chart_paths,
        capital_note,
    )

    print(f"PDF gespeichert: {pdf_path.resolve()}")
    for name, path in chart_paths.items():
        print(f"Chart {name}: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
