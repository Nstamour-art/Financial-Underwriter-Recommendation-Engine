try:
    from .prompt import build_prompt
except ImportError:
    from prompt import build_prompt

__all__ = ["build_prompt"]
