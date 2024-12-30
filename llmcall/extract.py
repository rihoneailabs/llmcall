from typing import List, Optional, Union

from pydantic import BaseModel
from .core import LLMConfig


def extract(
    text: str,
    output_schema: BaseModel,
    instructions: Optional[str] = None,
) -> Union[str, dict]:
    """Generate content using configured LLM."""
    pass  # Implementation here

