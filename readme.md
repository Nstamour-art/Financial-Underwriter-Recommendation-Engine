# Wealthsimple Underwriting & Recommendation Engine

This tool is a submission for the Wealthsimple AI Builder role and aims to automate recommendation of new products to potential clients based on transaction data rather than credit score.

The tool takes in transactions from available accounts, either as CSVs or from the Plaid Sandbox, then runs those through multiple stages ending with a API call to one of multiple available LLM providers. The results of this process will be a brief summary, a score and a recommendation of which product(s) the client should be recommended.

This process aims to make Wealthsimple more accessible to clients with limited credit history while at the same time reducing the amount of staffing required to review large complex and unstructured data from various sources.

Finally, if rejected, the system will ask for human intervention since, while not illegal, when assessing financial products for clients, it is uncontionable to let an AI make the final call on a rejection. Human judgement is best in these scenarios; that allows for sympathy and compassion as maybe the user used to have financial trouble, due to medical illness, but has since turned it around.

## Data Flow

For this tool the data flow will start as either account statements, provided as CSV files, or as a connection to Plaid. This will move into a cleaning and organization stage for CSV ingestion as CSVs from various institutions will be radically different. When ingesting via Plaid, the tool will use JSON and so will skip this stage and move to using NER and a local LLM to sort through transactions and add Category information. This stage will tally up NSF or Bank Fees, create a in/out calculation for all accounts and more. Following this, the tool will package the data into a dataclass to be used for generating the LLM API system prompt. The LLM will return a JSON with the recommended products, the client score and a summary of why it made that decision. If the LLM rejects the client, there will be opportunity for the user to review that decision and update review with a decision reasoning.
