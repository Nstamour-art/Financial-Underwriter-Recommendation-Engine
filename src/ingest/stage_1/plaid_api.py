import json
import os
import time
from datetime import date, timedelta
import plaid
from dotenv import load_dotenv
from plaid.api import plaid_api
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.products import Products
from plaid.model.sandbox_public_token_create_request import SandboxPublicTokenCreateRequest
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.transactions_get_request_options import TransactionsGetRequestOptions

load_dotenv()

class PlaidAPI:

    def __init__(self):
        self.client = self._build_client()
    
    def _load_sandbox_users(self) -> list[tuple[str, str, str]]:
        """
        Load sandbox test users from environment variables.
        Each user is defined by:
        PLAID_USER_<N>_USERNAME  — the override_username set in the Plaid Dashboard
        PLAID_USER_<N>_LABEL     — a human-readable label for logging/output
        All users share PLAID_INSTITUTION_ID unless you add per-user institution vars.
        """
        institution_id = os.getenv("PLAID_INSTITUTION_ID", "ins_109508")
        users = []
        for n in range(1, 4):  # slots 1, 2, 3
            username = os.getenv(f"PLAID_USER_{n}_USERNAME", "").strip()
            label = os.getenv(f"PLAID_USER_{n}_LABEL", f"User {n}").strip()
            if username:
                users.append((institution_id, username, label))
        if not users:
            raise RuntimeError(
                "No sandbox users configured. Set PLAID_USER_1_USERNAME (and optionally "
                "PLAID_USER_2_USERNAME, PLAID_USER_3_USERNAME) in your .env file."
            )
        return users


    def _build_client(self) -> plaid_api.PlaidApi:
        client_id = os.environ["PLAID_CLIENT_ID"]
        secret = os.environ["PLAID_SECRET"]
        env_name = os.getenv("PLAID_ENV", "sandbox").lower()

        env_map = {
            "sandbox":     plaid.Environment.Sandbox,
            "production":  plaid.Environment.Production,
        }
        host = env_map.get(env_name, plaid.Environment.Sandbox)

        configuration = plaid.Configuration(
            host=host,
            api_key={"clientId": client_id, "secret": secret},
        )
        api_client = plaid.ApiClient(configuration)
        return plaid_api.PlaidApi(api_client)


    def create_sandbox_access_token(
        self,
        client: plaid_api.PlaidApi,
        institution_id: str,
        override_username: str = "user_good",
        override_password: str = "pass_good",
    ) -> str:
        """Create a sandbox public token and exchange it for an access token."""
        pt_request = SandboxPublicTokenCreateRequest(
            institution_id=institution_id,
            initial_products=[Products("transactions")],
            options={
                "override_username": override_username,
                "override_password": override_password,
            },
        )
        pt_response = client.sandbox_public_token_create(pt_request)
        public_token = pt_response["public_token"]

        exchange_request = ItemPublicTokenExchangeRequest(public_token=public_token)
        exchange_response = client.item_public_token_exchange(exchange_request)
        return exchange_response["access_token"]


    def get_item_data(
        self,
        client: plaid_api.PlaidApi,
        access_token: str,
        start_date: date,
        end_date: date,
        max_count: int = 500,
        max_retries: int = 6,
        retry_delay: float = 5.0,
    ) -> dict:
        """
        Fetch all accounts and transactions for an item over the given date range.
        Handles pagination and retries on PRODUCT_NOT_READY.

        Returns:
            {
                "accounts": [<account dict>, ...],
                "transactions_by_account": {
                    "<account_id>": [<transaction dict>, ...],
                    ...
                }
            }
        """
        all_transactions: list[dict] = []
        accounts: list[dict] = []
        offset = 0
        response = None

        while True:
            options = TransactionsGetRequestOptions(
                count=min(500, max_count - len(all_transactions)),
                offset=offset,
            )
            request = TransactionsGetRequest(
                access_token=access_token,
                start_date=start_date,
                end_date=end_date,
                options=options,
            )

            for attempt in range(1, max_retries + 1):
                try:
                    response = client.transactions_get(request)
                    break
                except plaid.ApiException as e:
                    body = json.loads(e.body) if isinstance(e.body, str) else e.body
                    if body and body.get("error_code") == "PRODUCT_NOT_READY" and attempt < max_retries:
                        print(f"  -> PRODUCT_NOT_READY, retrying in {retry_delay}s "
                                f"(attempt {attempt}/{max_retries})")
                        time.sleep(retry_delay)
                    else:
                        raise

            assert response is not None  # assigned by the retry loop above (raises on all failures)
            # Accounts are the same on every page; capture once from the first response.
            if not accounts:
                accounts = [a.to_dict() for a in response["accounts"]]

            all_transactions.extend(t.to_dict() for t in response["transactions"])

            total_available = response["total_transactions"]
            if len(all_transactions) >= total_available or len(all_transactions) >= max_count:
                break
            offset = len(all_transactions)

        # Group transactions by account_id.
        transactions_by_account: dict[str, list[dict]] = {
            a["account_id"]: [] for a in accounts
        }
        for txn in all_transactions:
            aid = txn["account_id"]
            transactions_by_account.setdefault(aid, []).append(txn)

        return {"accounts": accounts, "transactions_by_account": transactions_by_account}


    def fetch_all_sandbox_transactions(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> dict[str, dict]:
        """
        Create sandbox access tokens for each test user, fetch all their accounts,
        and return transactions grouped by account.

        Returns:
            {
                "<user label>": {
                    "accounts": [...],
                    "transactions_by_account": {"<account_id>": [...], ...}
                },
                ...
            }
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=90)

        client = self._build_client()
        results: dict[str, dict] = {}

        for institution_id, username, label in self._load_sandbox_users():
            print(f"Fetching data for: {label} ({username})")
            try:
                access_token = self.create_sandbox_access_token(client, institution_id, username)
                data = self.get_item_data(client, access_token, start_date, end_date)
                results[label] = data

                total_txns = sum(len(t) for t in data["transactions_by_account"].values())
                print(f"  -> {len(data['accounts'])} accounts, {total_txns} transactions")
                for acct in data["accounts"]:
                    acct_txns = len(data["transactions_by_account"].get(acct["account_id"], []))
                    print(f"     [{acct['type']}] {acct['name']} — {acct_txns} transactions")
            except plaid.ApiException as e:
                print(f"  -> Plaid API error: {e.body}")
                results[label] = {"accounts": [], "transactions_by_account": {}}

        return results


if __name__ == "__main__":
    from pprint import pprint

    # Sandbox custom users generate transactions relative to today, not a fixed date.
    end = date.today()
    start = end - timedelta(days=360)

    api = PlaidAPI()
    
    all_data = api.fetch_all_sandbox_transactions(start_date=start, end_date=end)

    for user_label, data in all_data.items():
        print(f"\n{'='*60}")
        print(f"USER: {user_label}")
        for acct in data["accounts"]:
            pprint(acct)
            pprint(data["transactions_by_account"].get(acct["account_id"], []))
