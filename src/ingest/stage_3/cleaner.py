"""
Stage 3 – Transaction Memo Cleaner

Three-stage pipeline that normalises raw bank/brokerage transaction memos into
clean merchant names suitable for downstream LLM analysis:

  1. spaCy  – regex noise stripping + NLP-guided token filtering (always runs)
  2. BERT   – NER-based ORG entity extraction via dslim/bert-base-NER (always runs)
  3. Qwen 3 – LLM normalisation via llama_cpp (runs when BERT confidence < threshold)

All models are loaded lazily on first use.
"""

import re
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import spacy
from huggingface_hub import hf_hub_download
from transformers import pipeline as hf_pipeline
from llama_cpp import Llama

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from custom_dataclasses.user_data import User, Transaction


# ---------------------------------------------------------------------------
# Stage 1 helpers – spaCy / regex noise stripping
# ---------------------------------------------------------------------------

# Patterns applied left-to-right; order matters (most specific first).
_NOISE_PATTERNS = [
    r'\*[A-Z0-9]+',                                       # Amazon style: *AB12CD
    r'#\s*\w+',                                           # store/ref numbers: #1234
    r'\bREF:?\s*\w+',                                     # REF: ABC123
    r'\b(TXN|IDP|POS|CHQ|ACH|PYMT|PMT)\b',               # banking prefixes
    r'\b(PURCHASE|PAYMENT|TRANSFER|DEPOSIT|WITHDRAWAL)\b',
    r'\b[A-Z0-9]{10,}\b',                                 # long alphanumeric codes
    r'\b\d{6,}\b',                                        # long pure-numeric codes
    r'\b(BC|AB|ON|QC|NS|NB|MB|SK|PE|NL|NT|NU|YT|CA)\b',  # Canadian provinces/country
    r'\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b',                     # Canadian postal codes
    r'\b\d{2,4}[-/]\d{2}[-/]\d{2,4}\b',                  # embedded dates
    r'\$\d+\.?\d*',                                       # dollar amounts
    r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b',                  # phone numbers
    r'\s{2,}',                                            # collapse whitespace (last)
]

_NOISE_RES = [re.compile(p, re.IGNORECASE) for p in _NOISE_PATTERNS]


# ---------------------------------------------------------------------------
# Known banking term normalizations (checked before any ML runs)
# ---------------------------------------------------------------------------
# Ordered most-specific first so broader patterns don't shadow narrower ones.

_KNOWN_NORMALIZATIONS = [
    (re.compile(r'\bINTERAC[\s\-]*E[\s\-]*TR[FS]\b', re.I), 'e-Transfer'),
    (re.compile(r'\bINTERAC\b',                              re.I), 'e-Transfer'),
    (re.compile(r'\bE[\s\-]*TRANSFER\b',                     re.I), 'e-Transfer'),
    (re.compile(r'\bNSF\s+(?:FEE|CHARGE)\b',                 re.I), 'NSF Fee'),
    (re.compile(r'\bOVERDRAFT\s+(?:INTEREST|FEE|CHARGE)\b',  re.I), 'Overdraft Fee'),
    (re.compile(r'\bMONTHLY\s+(?:ACCOUNT\s+)?FEE\b',         re.I), 'Monthly Fee'),
    (re.compile(r'\bANNUAL\s+FEE\b',                         re.I), 'Annual Fee'),
    (re.compile(r'\bSERVICE\s+(?:FEE|CHARGE)\b',             re.I), 'Service Charge'),
    (re.compile(r'\bINTEREST\s+CHARGED\b',                   re.I), 'Interest Charge'),
    (re.compile(r'\bDIRECT\s+DEPOSIT\b',                     re.I), 'Direct Deposit'),
    (re.compile(r'\bPRE[\s\-]AUTH(?:ORIZED)?\b',             re.I), 'Pre-Authorized Debit'),
    (re.compile(r'\bATM\s+(?:WITHDRAWAL|FEE)\b',             re.I), 'ATM Withdrawal'),
]


def _check_known_normalization(text: str) -> Optional[str]:
    """Return a canonical label if the memo matches a known banking pattern."""
    for pattern, label in _KNOWN_NORMALIZATIONS:
        if pattern.search(text):
            return label
    return None


def _smart_title(text: str) -> str:
    """
    Title-case each word, but preserve all-caps abbreviations (NSF, ATM, etc.)
    which would be mangled by str.title().
    """
    def _cap_word(w: str) -> str:
        # Keep short all-caps words as-is (acronyms: NSF, ATM, BMO …)
        if w.isupper() and len(w) <= 5:
            return w
        return w.capitalize()
    return " ".join(_cap_word(w) for w in text.split())


def _apply_noise_strip(text: str) -> str:
    s = text
    for rx in _NOISE_RES[:-1]:
        s = rx.sub(' ', s)
    return _NOISE_RES[-1].sub(' ', s).strip()


def _spacy_clean(nlp, text: str) -> str:
    """
    Strip noise tokens, then use spaCy NER/POS to surface the most meaningful
    remaining tokens. Returns spaCy's best ORG entity if one is found, otherwise
    returns proper-noun / noun tokens joined as a phrase.
    """
    stripped = _apply_noise_strip(text)
    if not stripped:
        return text.strip()

    doc = nlp(stripped)

    # Prefer spaCy ORG entities — take the longest one.
    orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]
    if orgs:
        return max(orgs, key=len)

    # Fall back to meaningful POS tags (proper nouns, nouns; skip stopwords/punct)
    kept = [
        t.text for t in doc
        if not t.is_stop and not t.is_punct and t.pos_ in ("PROPN", "NOUN", "ADJ")
    ]
    return " ".join(kept) if kept else stripped


# ---------------------------------------------------------------------------
# Stage 2 helpers – BERT NER
# ---------------------------------------------------------------------------

def _bert_extract_org(
    ner_pipeline,
    text: str,
    threshold: float,
) -> Tuple[Optional[str], float]:
    """
    Return (entity_text, score) for the highest-confidence ORG entity, or
    (None, 0.0) if none exceed the threshold.
    """
    if not text.strip():
        return None, 0.0
    try:
        entities = ner_pipeline(text)
    except Exception:
        return None, 0.0

    orgs = [(e["word"], float(e["score"])) for e in entities if e["entity_group"] == "ORG"]
    if not orgs:
        return None, 0.0

    best_word, best_score = max(orgs, key=lambda x: x[1])
    if best_score >= threshold:
        return best_word.strip(), best_score
    return None, best_score


# ---------------------------------------------------------------------------
# Stage 3 helpers – Qwen 3 via llama_cpp
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Qwen 3 model download helper
# ---------------------------------------------------------------------------

# cleaner.py lives at src/ingest/stage_3/ → project root is three levels up
_PROJECT_ROOT  = Path(__file__).resolve().parents[3]
_MODELS_DIR    = _PROJECT_ROOT / "models"
_GGUF_FILENAME = "Qwen3-8B-Q4_K_M.gguf"
_HF_REPO_ID    = "Qwen/Qwen3-8B-GGUF"


def download_qwen3_model(models_dir: Path = _MODELS_DIR) -> Path:
    """
    Return the local path to the Qwen3-8B-Q4_K_M.gguf model, downloading it
    from Hugging Face Hub into *models_dir* if it is not already present.

    The download uses ``huggingface_hub.hf_hub_download`` which shows a
    progress bar and resumes interrupted downloads automatically.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / _GGUF_FILENAME
    if dest.exists():
        return dest

    print(f"Downloading {_GGUF_FILENAME} from {_HF_REPO_ID} → {dest} …")
    hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename=_GGUF_FILENAME,
        local_dir=str(models_dir),
    )
    return dest


# ---------------------------------------------------------------------------
# Stage 3 helpers – Qwen 3 via llama_cpp
# ---------------------------------------------------------------------------

_LLM_SYSTEM = (
    "/no_think\n"
    "You are a financial data cleaner. Extract the merchant or payee name from "
    "a raw bank transaction memo.\n"
    "Rules:\n"
    "- Reply with ONLY the merchant name — no explanation, no punctuation\n"
    "- Use title case (e.g. 'STARBUCKS' → 'Starbucks')\n"
    "- Remove store numbers, city names, province codes, and transaction IDs\n"
    "- For fees or bank charges use a concise label (e.g. 'NSF Fee', 'Monthly Fee')\n"
    "- For e-transfers or interac use 'e-Transfer'\n"
)

_LLM_EXAMPLES: List[Tuple[str, str, str]] = [
    # (raw, spacy_cleaned, merchant)
    ("STARBUCKS STORE #1234 VANCOUVER BC",  "Starbucks",          "Starbucks"),
    ("AMZN MKTP CA*A1B2C3D4",              "AMZN MKTP",          "Amazon"),
    ("NETFLIX.COM 866-579-7172 CA",         "Netflix.com",        "Netflix"),
    ("SHELL OIL 0123456789 RICHMOND BC",    "Shell Oil",          "Shell"),
    ("NSF FEE CHARGED BY BANK",             "NSF Fee Bank",       "NSF Fee"),
    ("INTERAC E-TRF 1234567890",            "INTERAC E-TRF",      "e-Transfer"),
    ("PAYROLL DIRECT DEPOSIT ACME CORP",    "ACME CORP",          "Acme Corp"),
]


def _strip_think_tags(text: str) -> str:
    """Remove Qwen 3 <think>...</think> blocks from output."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def _llm_extract_merchant(llm: Llama, raw: str, spacy_cleaned: str) -> str:
    messages = [{"role": "system", "content": _LLM_SYSTEM}]
    for raw_ex, spacy_ex, clean_ex in _LLM_EXAMPLES:
        messages.append({
            "role": "user",
            "content": f"Raw: {raw_ex}\nCleaned: {spacy_ex}\nMerchant:",
        })
        messages.append({"role": "assistant", "content": clean_ex})

    messages.append({
        "role": "user",
        "content": f"Raw: {raw}\nCleaned: {spacy_cleaned}\nMerchant:",
    })

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=24,
        temperature=0.0,
        stop=["\n", "<|im_end|>", "<|endoftext|>"],
    )
    raw_output = response["choices"][0]["message"]["content"]
    return _strip_think_tags(raw_output).strip()


# ---------------------------------------------------------------------------
# TransactionCleaner
# ---------------------------------------------------------------------------

class TransactionCleaner:
    """
    Normalises raw transaction memos into clean merchant names.

    Pipeline per memo:
        1. spaCy  – regex noise + NLP token filtering          (always)
        2. BERT   – ORG entity extraction (dslim/bert-base-NER)(always)
        3. Qwen 3 – LLM normalisation via llama_cpp            (when BERT < threshold)

    Parameters
    ----------
    llm_model_path : str or None
        Path to a Qwen 3 GGUF file.  If None, the LLM stage is skipped and
        the pipeline falls back to the spaCy result when BERT is not confident.
    n_ctx : int
        Context window for llama_cpp (default 512 is sufficient for memo cleaning).
    n_gpu_layers : int
        Number of model layers to offload to GPU (-1 = all).  0 = CPU only.
    ner_threshold : float
        Minimum BERT ORG confidence to accept without calling the LLM.
    verbose : bool
        Pass-through to llama_cpp Llama verbosity.
    """

    NER_MODEL   = "dslim/bert-base-NER"
    SPACY_MODEL = "en_core_web_sm"

    def __init__(
        self,
        llm_model_path: Optional[str] = None,
        n_ctx: int = 512,
        n_gpu_layers: int = 0,
        ner_threshold: float = 0.85,
        verbose: bool = False,
    ):
        self._llm_model_path = llm_model_path
        self._n_ctx          = n_ctx
        self._n_gpu_layers   = n_gpu_layers
        self._ner_threshold  = ner_threshold
        self._verbose        = verbose

        self._nlp: Optional[spacy.language.Language] = None
        self._ner = None   # HuggingFace NER pipeline
        self._llm: Optional[Llama] = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load_spacy(self) -> None:
        if self._nlp is not None:
            return
        try:
            self._nlp = spacy.load(self.SPACY_MODEL)
        except OSError:
            from spacy.cli import download as spacy_download
            print(f"Downloading spaCy model '{self.SPACY_MODEL}'...")
            spacy_download(self.SPACY_MODEL)
            self._nlp = spacy.load(self.SPACY_MODEL)

    def _load_ner(self) -> None:
        if self._ner is None:
            self._ner = hf_pipeline(
                "ner",
                model=self.NER_MODEL,
                aggregation_strategy="simple",
            )

    def _load_llm(self) -> None:
        if self._llm is not None:
            return
        if not self._llm_model_path:
            self._llm_model_path = str(download_qwen3_model())
        self._llm = Llama(
            model_path=self._llm_model_path,
            n_ctx=self._n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            verbose=self._verbose,
        )

    def _ensure_base_loaded(self) -> None:
        self._load_spacy()
        self._load_ner()

    # ------------------------------------------------------------------
    # Core cleaning logic
    # ------------------------------------------------------------------

    def clean_memo(self, memo: str) -> str:
        """
        Run the full three-stage pipeline on a single raw memo string.
        Returns the normalised merchant name.
        """
        if not memo or not memo.strip():
            return memo

        # Known banking patterns — deterministic, no ML needed
        known = _check_known_normalization(memo)
        if known:
            return known

        self._ensure_base_loaded()

        # Stage 1 – spaCy
        spacy_result = _spacy_clean(self._nlp, memo)

        # Stage 2 – BERT NER
        bert_result, _ = _bert_extract_org(
            self._ner,
            spacy_result or memo,
            self._ner_threshold,
        )
        if bert_result:
            return _smart_title(bert_result)

        # Stage 3 – Qwen 3 (only when BERT is not confident)
        if self._llm_model_path:
            self._load_llm()
            return _llm_extract_merchant(self._llm, memo, spacy_result)

        # Fallback: return best spaCy result
        return _smart_title(spacy_result) if spacy_result else _smart_title(memo.strip())

    # ------------------------------------------------------------------
    # Batch / User-level API
    # ------------------------------------------------------------------

    def clean_users(self, users: List[User]) -> List[User]:
        """
        Clean the ``cleaned_description`` field of every Transaction in the
        provided User objects.  Modifies transactions in-place and returns
        the same list.
        """
        self._ensure_base_loaded()
        for user in users:
            for account in user.accounts:
                for txn in account.transactions:
                    txn.cleaned_description = self.clean_memo(txn.memo)
        return users


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEST_MEMOS = [
        "STARBUCKS STORE #1234 VANCOUVER BC",
        "AMZN MKTP CA*A1B2C3D4",
        "NETFLIX.COM 866-579-7172 CA",
        "SHELL OIL 0123456789 RICHMOND BC",
        "INTERAC E-TRF 9876543210",
        "NSF FEE CHARGED BY BANK",
        "PAYROLL DIRECT DEPOSIT ACME CORP INC",
        "COSTCO WHOLESALE #0123 BURNABY BC V5H 1A1",
        "TIM HORTONS #4567",
        "BUY  VFV  50.000 SHS @ 120.500",
        "SELL ETF ABC123 20 SHS",
    ]

    # Model auto-downloads on first use via download_qwen3_model()
    cleaner = TransactionCleaner(verbose=False)

    print(f"{'Raw memo':<50}  {'Cleaned'}")
    print("-" * 70)
    for memo in TEST_MEMOS:
        cleaned = cleaner.clean_memo(memo)
        print(f"{memo:<50}  {cleaned}")
