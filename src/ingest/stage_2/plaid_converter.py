import os
import sys
from datetime import date, datetime
from decimal import Decimal
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from custom_dataclasses.user_data import User, Account, Transaction


def _parse_date(value) -> date:
    if isinstance(value, (date, datetime)):
        return value.date() if isinstance(value, datetime) else value
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d-%b-%Y", "%d-%m-%Y", "%m-%d-%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date: {s!r}")


class PlaidDataConverter:
    """Convert the raw dict output of PlaidAPI.fetch_all_sandbox_transactions() into User objects."""

    @staticmethod
    def convert(raw_data: Dict[str, Dict[str, Any]]) -> List[User]:
        users = []
        for user_label, data in raw_data.items():
            accounts = []
            for acct in data["accounts"]:
                balances = acct.get("balances") or {}

                def _dec(v) -> Optional[Decimal]:
                    return Decimal(str(v)) if v is not None else None

                _INV_CATEGORY: dict[str, str] = {
                    "buy":      "Investments",
                    "sell":     "Investments",
                    "cash":     "Investments",
                    "transfer": "Transfers",
                    "fee":      "Fees & Charges",
                    "dividend": "Income",
                    "interest": "Income",
                }

                transactions = []
                for txn in data["transactions_by_account"].get(acct["account_id"], []):
                    raw_amount = txn.get("amount", 0) or 0
                    # Plaid: positive = debit (money out). Our convention: negative = debit.
                    amount = Decimal(str(raw_amount)) * Decimal("-1")
                    txn_type = "credit" if amount >= 0 else "debit"

                    if txn.get("_source") == "investment":
                        inv_type = (txn.get("type") or "").lower()
                        memo = txn.get("name") or f"Investment {inv_type}".strip()
                        category = [_INV_CATEGORY.get(inv_type, "Investments")]
                    else:
                        pfc = txn.get("personal_finance_category") or {}
                        legacy = txn.get("category") or []
                        category = ([pfc["primary"]] if pfc.get("primary") else legacy) or []
                        memo = txn.get("name") or ""

                    transactions.append(Transaction(
                        date=_parse_date(txn["date"]),
                        account_id=acct["account_id"],
                        amount=amount,
                        memo=memo,
                        cleaned_description=txn.get("merchant_name") or memo,
                        category=category,
                        transaction_type=txn_type,
                        source="plaid",
                    ))

                accounts.append(Account(
                    account_id=acct["account_id"],
                    name=acct.get("name") or acct.get("official_name") or "",
                    type=acct.get("type", "other"),
                    subtype=acct.get("subtype"),
                    current_balance=_dec(balances.get("current")),
                    available_balance=_dec(balances.get("available")),
                    transactions=transactions,
                    credit_limit=_dec(balances.get("limit")),
                    currency=balances.get("iso_currency_code") or "CAD",
                ))

            users.append(User(user_id=user_label, name=None, accounts=accounts))

        return users
    


