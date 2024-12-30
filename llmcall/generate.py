from typing import List, Optional, Union

from pydantic import BaseModel
from .core import LLMConfig


def generate(
    prompt: str,
    output_schema: BaseModel,
    instructions: Optional[str] = None,
) -> Union[str, dict]:
    """Generate content using configured LLM."""
    pass  # Implementation here

def generate_decision(
    prompt: str,
    options: List[str],
    instructions: Optional[str] = None,
) -> str:
    """Generate a decision from a list of options."""
    pass  # Implementation here
