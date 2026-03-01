import hashlib
import os
import sys
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from custom_dataclasses.user_data import User, Account, Transaction
from custom_dataclasses.csv_input import CSVFileInput
from ingest.stage_2.column_identifier import ColumnIdentifier


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_DATE_FORMATS = [
    "%Y-%m-%d",     # 2026-01-15
    "%Y/%m/%d",     # 2026/01/15
    "%m/%d/%Y",     # 01/15/2026
    "%d-%b-%Y",     # 28-Feb-2026
    "%d-%m-%Y",     # 15-01-2026
    "%m-%d-%y",     # 02-14-26  (RRSP)
    "%d/%m/%Y",     # 15/01/2026
]


def _parse_date(value) -> date:
    if isinstance(value, (date, datetime)):
        return value.date() if isinstance(value, datetime) else value
    s = str(value).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date: {s!r}")


def _parse_amount(value) -> Decimal:
    """
    Parse a raw CSV amount string.
    - (value)  → negative  (bank credit/income notation)
    - -value   → negative  (debit/expense)
    - value    → positive
    """
    if pd.isna(value) or str(value).strip() == "":
        return Decimal("0")
    s = str(value).strip().replace(",", "").replace("$", "")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return Decimal(s)
    except InvalidOperation:
        return Decimal("0")


# ---------------------------------------------------------------------------
# Account type from filename
# ---------------------------------------------------------------------------

_FILENAME_TYPE_MAP = [
    (["chequing", "checking"],  ("depository", "checking")),
    (["saving"],                ("depository", "savings")),
    (["credit"],                ("credit",     "credit card")),
    (["rrsp"],                  ("investment", "rrsp")),
    (["tfsa"],                  ("investment", "tfsa")),
    (["rrif"],                  ("investment", "rrif")),
    (["mortgage"],              ("loan",       "mortgage")),
    (["loan"],                  ("loan",       "loan")),
]


def _account_type_from_stem(stem: str):
    normalized = stem.lower().replace("_", " ").replace("-", " ")
    for keywords, result in _FILENAME_TYPE_MAP:
        if any(kw in normalized for kw in keywords):
            return result
    return ("other", normalized.strip())


def _account_id(user_label: str, stem: str) -> str:
    raw = f"{user_label}::{stem}".lower()
    return hashlib.md5(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# CSVDataConverter
# ---------------------------------------------------------------------------

class CSVDataConverter:
    """
    Convert CSVFileInput objects (one per account file) into User objects.

    Expected input:
        {
            "user_debt_spiral": [
                CSVFileInput(filepath="...", df=<DataFrame>, account_type=("depository", "checking")),
                CSVFileInput(filepath="...", df=<DataFrame>),  # type inferred from filename
            ],
            ...
        }
    """

    @staticmethod
    def convert(
        csv_data: Dict[str, List[CSVFileInput]],
        currency: str = "CAD",
    ) -> List[User]:
        users = []
        for user_label, file_inputs in csv_data.items():
            accounts = [
                CSVDataConverter._convert_file(
                    filepath=fi.filepath,
                    df=fi.df,
                    user_label=user_label,
                    currency=currency,
                    account_type=fi.account_type,
                )
                for fi in file_inputs
            ]
            users.append(User(user_id=user_label, name=None, accounts=accounts))
        return users

    @staticmethod
    def _convert_file(
        filepath: str,
        df: pd.DataFrame,
        user_label: str,
        currency: str = "CAD",
        account_type: Optional[tuple] = None,
    ) -> Account:
        stem = Path(filepath).stem
        acct_type, subtype = account_type if account_type else _account_type_from_stem(stem)
        account_id = _account_id(user_label, stem)

        identified = ColumnIdentifier.identify(list(df.columns))
        date_col   = identified["date"]
        desc_col   = identified["description"]
        amount_col = identified["amount"]
        debit_col  = identified["debit"]
        credit_col = identified["credit"]
        shares_col = identified["shares"]
        price_col  = identified["price"]
        symbol_col = identified["symbol"]

        transactions = []
        for _, row in df.iterrows():
            txn_date = _parse_date(row[date_col]) if date_col else date.today()

            raw_desc = str(row[desc_col]).strip() if desc_col else ""
            if acct_type == "investment" and symbol_col:
                raw_desc = f"{raw_desc} {str(row[symbol_col]).strip()}".strip()

            if amount_col:
                amount = _parse_amount(row[amount_col])
            elif debit_col or credit_col:
                debit  = _parse_amount(row[debit_col])  if debit_col  else Decimal("0")
                credit = _parse_amount(row[credit_col]) if credit_col else Decimal("0")
                amount = credit - debit
            elif shares_col and price_col:
                shares = _parse_amount(row[shares_col])
                price  = _parse_amount(row[price_col])
                sign   = Decimal("-1") if "buy" in raw_desc.lower() else Decimal("1")
                amount = sign * shares * price
            else:
                amount = Decimal("0")

            transactions.append(Transaction(
                date=txn_date,
                account_id=account_id,
                amount=amount,
                memo=raw_desc,
                cleaned_description=raw_desc,
                category=[],
                transaction_type="credit" if amount >= 0 else "debit",
                source="csv",
            ))

        return Account(
            account_id=account_id,
            name=stem.replace("_", " "),
            type=acct_type,
            subtype=subtype,
            current_balance=None,
            available_balance=None,
            transactions=transactions,
            credit_limit=None,
            currency=currency,
        )


if __name__ == "__main__":
    from ingest.stage_1.csv_loader import CSVLoader

    base = Path("data/csv_users")
    csv_data: Dict[str, List[CSVFileInput]] = {}

    for user_dir in sorted(base.iterdir()):
        if not user_dir.is_dir():
            continue
        csv_data[user_dir.name] = [
            CSVFileInput(filepath=str(f), df=CSVLoader(str(f)).load_csv())
            for f in sorted(user_dir.glob("*.csv"))
        ]

    users = CSVDataConverter.convert(csv_data)
    for user in users:
        print(f"\nUser: {user.user_id}")
        for acct in user.accounts:
            print(f"  [{acct.type}/{acct.subtype}] {acct.name} "
                    f"(id={acct.account_id}) — {len(acct.transactions)} transactions")
            for txn in acct.transactions[:3]:
                print(f"    {txn.date}  {txn.transaction_type:6}  {txn.amount:>10}  {txn.memo}")
