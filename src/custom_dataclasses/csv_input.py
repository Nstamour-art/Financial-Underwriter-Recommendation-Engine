from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


@dataclass
class CSVFileInput:
    """
    Bundles a CSV file with its parsed DataFrame and an optional account type
    override. Intended to be constructed by a UI or loader before passing to
    CSVDataConverter.

    account_type: (type, subtype) tuple, e.g. ("depository", "checking").
    If None, account type is inferred from the filename stem.
    """
    filepath: str
    df: pd.DataFrame
    account_type: Optional[tuple] = None  # (type, subtype)
