from dataclasses import dataclass
from typing import Literal, Optional, List


ProductType = Literal["savings", "investing", "crypto", "credit"]


@dataclass(frozen=True)
class Product:
    name: str
    type: ProductType
    description: str
    # Guidance explaining the ideal client profile; passed verbatim to the LLM.
    recommendation_notes: str
    # Minimum estimated annual income (CAD) to recommend this product.
    # None means no income threshold applies.
    min_annual_income: Optional[int] = None

    def is_eligible(self, annual_income_est: Optional[float]) -> bool:
        """Return True if the estimated annual income meets this product's threshold."""
        if self.min_annual_income is None:
            return True
        if annual_income_est is None:
            return False
        return annual_income_est >= self.min_annual_income

    def prompt_line(self) -> str:
        """Compact two-line entry for use inside an LLM system prompt."""
        income_req = (
            f"min income CAD ${self.min_annual_income:,}/yr"
            if self.min_annual_income
            else "no income requirement"
        )
        return (
            f"- {self.name}: {self.description} ({income_req})\n"
            f"  Guidance: {self.recommendation_notes}"
        )

@dataclass(frozen=True)
class ProductCatalog:
    """A simple wrapper around a list of Products, with helper methods for LLM prompting and filtering."""

    products: List[Product]

    def by_type(self, product_type: "ProductType") -> List[Product]:
        """Return all products of a given type."""
        return [p for p in self.products if p.type == product_type]


    def eligible_for(self, annual_income_est: Optional[float]) -> List[Product]:
        """Return products the client is income-eligible for."""
        return [p for p in self.products if p.is_eligible(annual_income_est)]


    def products_for_prompt(self, products: Optional[List[Product]] = None) -> str:
        """
        Compact multi-line string listing products for the LLM system prompt.
        Each product renders as a two-line block via Product.prompt_line().
        """
        return "\n".join(p.prompt_line() for p in (products or self.products))