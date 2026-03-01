try:
    from .llm_orchestrator import UnderwritingOrchestrator
    from .products import PRODUCTS
except ImportError:
    from llm_orchestrator import UnderwritingOrchestrator
    from products import PRODUCTS

__all__ = [
    "UnderwritingOrchestrator",
    "PRODUCTS",
]
