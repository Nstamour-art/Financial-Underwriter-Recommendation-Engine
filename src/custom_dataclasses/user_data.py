from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import List, Literal, Optional

@dataclass
class Transaction:
    date: date
    account_id: str
    amount: Decimal # Positive for credits, negative for debits
    memo: str # the raw transaction description
    cleaned_description: str # a cleaned-up description (e.g. "STARBUCKS" instead of "STARBUCKS STORE #1234")
    category: List[str] # e.g. ["Food and Dining"]
    transaction_type: Literal["debit", "credit"]
    source: Literal["plaid", "csv"]
    subcategory: Optional[str] = None # e.g. "Restaurants, fast food, and takeout"
    
@dataclass
class Account:
    account_id: str
    name: str
    type: str # e.g. "depository", "credit", "loan"
    subtype: Optional[str] # e.g. "checking", "savings", "credit card"
    current_balance: Optional[Decimal]
    available_balance: Optional[Decimal] # may be None for credit accounts
    transactions: List[Transaction]
    credit_limit: Optional[Decimal] = None # only for credit accounts
    apr: Optional[Decimal] = None # only for credit accounts
    currency: str = "CAD" # ISO currency code, default to CAD since we're focused on Canadian users
    
@dataclass
class User:
    user_id: str
    name: Optional[str] = None
    accounts: Optional[List[Account]] = None