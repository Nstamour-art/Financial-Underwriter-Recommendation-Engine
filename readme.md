# Financial Underwriter & Recommendation Engine

A full-stack AI pipeline that ingests raw bank transaction data, cleans and classifies it using a cascade of NLP models, and produces a credit-style underwriting decision with product recommendations - all without relying on a traditional credit bureau score.

Built as an interactive Streamlit application with live progress tracking, Altair/Plotly visualisations, and a employee review layer for rejected applicants.

## Why This Project

Credit underwriting is a legacy system. If a person were designing it today, with the technology available, they wouldn't reduce your financial identity to a single bureau score. They would look at what you actually do with your money and leverage new technology to make that faster and more personal.

I built this because I've lived there. In 2022, a severe medical condition left me unable to work for two years. I fell behind on payments, not because I was irresponsible, but because life happened. My credit score collapsed. Rebuilding it took years of effort and consistency, even though my underlying financial behaviour was sound. That experience made the gap in the system personal for me. I realized that creditworthy people are routinely excluded by a metric that can't distinguish hardship from risk.

This project is my answer to the question: *what would underwriting look like if it were built today?* It replaces the single-number with a full behavioural analysis pipeline — ingesting raw transactions, cleaning and classifying them through a cascade of NLP models, and producing a structured recommendation that a human advisor can act on.

This project combines the best of both worlds.

- The AI handles the complexity that makes manual statement review impractical at scale.
- The human retains authority over the consequential outcome when a score is too low.

I chose this because context like recovering from illness or irregular but legitimate income is exactly the kind of judgment that shouldn't be automated away, but is too complex to see when a human is reviewing this alone.

The system is designed to act not just as a demo, but as a prototype for a solution to a real problem. In the real world it could function to help with financial decisions with real constraints (offline model fallbacks, confidence calibration, PII-free logging, human override workflows). It's the kind of thing I'd want to exist if I were sitting across from an advisor, hoping someone would look beyond the number.

## How It Works

### Pipeline Overview

```text
                    Bank data (CSV / Plaid)
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                    Stage 1 - Ingest                        │
│              CSV loader or Plaid API fetch                 │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│                    Stage 2 - Convert                       │
│    Normalise to User / Account / Transaction dataclasses   │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│       Stage 3a - Memo Cleaning  (TransactionCleaner)       │
│                                                            │
│       Fast path:  no noise flags → smart_title()           │
│      spaCy  → regex noise strip + NLP token filter         │
│     BERT   → dslim/bert-base-NER ORG entity extraction     │
│     Qwen 3 → llama_cpp LLM fallback when BERT < thresh     │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│     Stage 3b - Categorization  (TransactionCategorizer)    │
│                                                            │
│   Zero-shot NLI: MoritzLaurer/deberta-v3-small-zeroshot    │
│         Two-pass: top-level category → subcategory         │
│          Pre-ML lookup table for common patterns           │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│      Process - Underwriting  (UnderwritingOrchestrator)    │
│                                                            │
│    Financial summary computed from cleaned transactions    │
│  Prompt sent to Anthropic claude-sonnet-4-6 or GPT-4o-mini │
│    Returns: score (300–900), decision, summary, products   │
│   Confidence = 60% LLM + 40% heuristic (data completeness) │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│                  Audit Trail (SQLite)                      │
│                                                            │
│       Every decision logged - score, confidence,           │
│       input hash (no PII), model provenance                │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│                      Streamlit UI                          │
│  Overview · Spending · Products · Transactions · Review    │
└────────────────────────────────────────────────────────────┘
```

## Stage Breakdown

The following section will breakdown the ML stages in the data ingest and output pipeline.

### Stage 3a - Memo Cleaning

**The Problem**: Raw bank memos are very noisy and difficult to understand. Example: `AMZN MKTP CA*A1B2C3D4`, `STARBUCKS STORE #1234 VANCOUVER BC 4520`.

**The Solution**: The cleaning stage normalises these into usable merchant names through a three-tier cascade:

1. **Pythonic Quick Path** - A lookup table handles common banking terms instantly (e-Transfer, NSF Fee, Overdraft Fee, Direct Deposit, Infufficient Funds, etc.) and a "dirty flag" check identifies memos that are already clean merchant names in ALLCAPS (`EQUIFAX`, `NOBU RES`). If either occurs the tool simply title-cases them, and skips the ML stack entirely.

2. **ML Stage 1 - spaCy NER** - Regex strips reference numbers, province codes, postal codes, phone numbers, and dollar amounts. The remaining tokens are processed for NER and POS tags; the longest ORG entity wins, with geographic entities (cities, provinces) excluded from the fallback.

3. **ML Stage 2 - BERT NER** (`dslim/bert-base-NER`) - This sage extracts named organisation entities from the spaCy-cleaned text with a configurable confidence threshold (default 0.85). Accepted results are returned immediately.

4. **ML Stage 3 - Qwen 3 8B** (`Qwen3-8B-Q4_K_M.gguf` via `llama_cpp`) - Stage 3 uses a local LLM only when BERT confidence falls below the threshold. A small prompt is used along with the results form the previous stages. The model auto-downloads from Hugging Face Hub on first use; runs fully offline thereafter.

### Stage 3b - Categorization

Each processed transaction is classified into a two-level taxonomy (9 top-level categories, e.g. Housing, Food and Dining, Transportation; plus subcategories) using zero-shot NLI inference:

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Two-pass:** Stage first classifies the sub category, then derives the top level from that.
- **Large payments boost:** Transactions with `amount > 200` have "Income" surfaced first in the candidate list.
- **Pre-ML lookup:** Common patterns like (PAYROLL, MORTGAGE, etc.) bypass the model entirely.
- **Batch inference** Batches the users' transactions for throughput.

### Underwriting & Recommendation

After cleaning and categorisation, a compact financial summary is derived programmatically from the transaction data - monthly average income, expenses, net cash flow, spending by category, credit utilisation, NSF/overdraft/payday loan counts, and account balances. This structured summary (not raw transaction text) is sent to the LLM for low token counts and also for security.

The system prompt instructs the model to return a strict JSON schema:

```json
{
  "score": 720,
  "decision": "approved",
  "confidence": 0.84,
  "summary": "...",
  "rejection_reason": null,
  "recommended_products": ["Wealthsimple Cash", "Wealthsimple Credit Card"],
  "top_product_reason": "..."
}
```

**Confidence calibration** blends the LLM's self-reported confidence (60%) with a heuristic score (40%) derived from data completeness signals: whether income is present, how many months of history are available, whether NSF events occurred, etc. This prevents the model from expressing false certainty on thin data.

**Providers:** Anthropic (`claude-sonnet-4-6`) is preferred; OpenAI (`gpt-4o-mini`) is the fallback. The provider is auto-detected from environment keys or explicitly selected in the UI.

---

### Human Review

When an applicant is rejected, the system does not treat the decision as final. The Decision Review tab surfaces the LLM's reasoning, confidence score, and data signals, and gives the advisor the ability to override the decision with written justification. This reflects a core design principle: AI handles data complexity, humans handle consequential outcomes - especially in cases where context matters (recovering from past hardship, irregular but legitimate income, etc.).

---

### Audit Trail

Every pipeline run writes an immutable record to a local SQLite database (`audit_logs/audit.sqlite`) containing:

- Timestamp, session ID, user ID
- SHA-256 hash of the input data (no PII stored directly)
- Score, decision, confidence (LLM + heuristic + combined)
- Recommended products
- Data completeness signals
- Model provenance (provider, NER model, categoriser)

This makes every decision traceable and reproducible for compliance review.

---

## Data Sources

**Plaid Sandbox** - The app connects to Plaid's sandbox environment to fetch multi-user transaction histories with realistic data across chequing, savings, and credit accounts. Configurable date range and per-user selection from the sidebar.

**CSV Upload** - Upload one or more CSV statement files exported from any Canadian financial institution. The pipeline auto-detects column roles (date, amount, description, transaction type) using `sentence-transformers/all-MiniLM-L6-v2` semantic similarity against anchor phrases, with a fallback to heuristic header matching. Multiple files can be grouped into accounts (e.g. combine 12 monthly statements into one history).

---

## Product Catalogue

The engine recommends from a curated set of financial products, each with eligibility rules enforced both in the system prompt and in the product catalogue:

| Product | Type | Key Eligibility |
| --- | --- | --- |
| Wealthsimple Cash | Savings / HISA | None - broadly available |
| Wealthsimple TFSA | Investment | None |
| Wealthsimple RRSP | Investment | None |
| Wealthsimple Credit Card | Credit | ≥ CAD $60,000 annual income |
| Wealthsimple Crypto | Investment | Positive monthly net, no NSF/overdraft |

---

## Streamlit UI

The app is structured around five tabs rendered after a pipeline run:

- **Overview** - score gauge, decision badge, top recommended product card, cash flow line chart, and LLM summary
- **Spending** - monthly stacked bar chart by category and pie chart of total spend distribution
- **Products** - full product catalogue grid, recommended products highlighted
- **Transactions** - filterable, searchable table of all cleaned and categorised transactions
- **Decision Review** - confidence breakdown, data signals, LLM reasoning, and human override form

Model warmup runs in a background thread at startup (spaCy, BERT NER, DeBERTa) so models are cached and ready before the user clicks Run.

---

## Tech Stack

| Layer | Technology |
| --- | --- |
| UI | Streamlit, Altair, Plotly |
| NLP - cleaning | spaCy `en_core_web_sm`, `dslim/bert-base-NER` (HuggingFace Transformers) |
| NLP - categorisation | `MoritzLaurer/deberta-v3-small-zeroshot-v1.1-all-33` |
| NLP - column detection | `sentence-transformers/all-MiniLM-L6-v2` |
| Local LLM | Qwen 3 8B Q4_K_M via `llama_cpp` |
| Cloud LLM | Anthropic claude-sonnet-4-6 / OpenAI gpt-4o-mini |
| Bank data | Plaid API (sandbox) |
| Audit | SQLite via `sqlite3` |
| Language | Python 3.11+ |

---

## Setup

```bash
# Clone and install
git clone <repo>
cd Financial-Underwriter-Recommendation-Engine
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Configure environment variables
cp .env.example .env
# Fill in: ANTHROPIC_API_KEY or OPENAI_API_KEY, PLAID_CLIENT_ID, PLAID_SECRET and HF_KEY for Hugging Face Inference

# Run
streamlit run main.py
```

> [ !Note ]
> To use the Plaid API connection you need to set up your Sandbox users and add them to the .env under the variables: `PLAID_USER_(N)_USERNAME`. The system supports 3 Plaid Sandbox users and their JSON can be found in `data/plaid_users/`.

The Qwen 3 GGUF model (~5 GB) is downloaded automatically from Hugging Face Hub into `models/` on first use if the LLM cleaning stage is reached. All other models are downloaded by HuggingFace Transformers on first inference.

---

## Project Structure

```text
.streamlit/
    config.toml                          ← Streamlit theme settings
static/
    imgs/                                ← images and videos for the application
    fonts/                               ← contains system fonts for Streamlit
src/
  orchestrator.py                        ← end-to-end pipeline entry point
  custom_dataclasses/
        user_data.py                     ← User, Account, Transaction
        csv_input.py                     ← CSVFileInput
        product.py                       ← Product, ProductCatalog
  ingest/
    stage_1/   
        csv_loader.py                    ← CSVLoader class to load and return Pandas dataframes
        plaid_api.py                     ← PlaidAPI class to handshake with Plaid network
    stage_2/   
        csv_converter.py                 ← CSV converter to identify the columns and prepare for cleaning
        plaid_converter.py               ← Plaid data converter to sort and normalize transactions
    stage_3/   
        cleaner.py  
        categorizer.py
  process/
    products.py                          ← ProductCatalog instance
    prompts/    
        prompt.py                        ← build_prompt()
    llm/        
        anthropic_api.py                 ← handles connection with Anthropic API
        open_ai_api.py                   ← handles connection with Open AI API
    llm_orchestrator.py
    audit.py                             ← SQLite audit trail
  ui/
    app.py                               ← all Streamlit rendering
    styles.py                            ← centralised colours and CSS
main.py                                  ← Streamlit entry point
```
## Demonstartion Video
[Follow this link to view the demonstration video](https://youtu.be/x-Y0IqfDQKY)

## Screenshots

![Score Page showing the score for a middle income Plaid Sandbox user.](resources\screenshots\ScoreScreen.png "Score Page")
![Spending page showing the trends for this user.](resources\screenshots\SpendingTrends.png "Spending Page")
![Products Recommendation Page.](resources\screenshots\Products.png "Products Page")
![Transactions Table with filtering.](resources\screenshots\Transactions.png "Transaction Page")
![Employee backend with Overide feature.](resources\screenshots\EmployeeOversight.png "Decision Review Page")
