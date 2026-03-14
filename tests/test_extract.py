from typing import Optional
from unittest.mock import patch

from pydantic import BaseModel

from llmcall import extract


def test_extract_from_unstructured_text(mock_cfg, mock_text_response):
    text = ("We are looking for qualified Uber Drivers in Cape Town. If you are interested, "
            "please contact us at 021 123 4567. Salary is R5000 per month (negotiable).")

    class JobAd(BaseModel):
        title: str
        location: str
        contact: Optional[str]
        salary: Optional[str]

    with patch("llmcall.extract.get_config", return_value=mock_cfg), patch(
        "llmcall.extract.completion",
        return_value=mock_text_response(
            '{"title":"Uber Driver","location":"Cape Town","contact":"021 123 4567","salary":"R5000 per month"}'
        ),
    ):
        result = extract(text=text, output_schema=JobAd)
    assert result.title.lower() == "uber driver"
    assert result.location.lower() == "cape town"
    assert result.contact == "021 123 4567"
    assert "R5000" in result.salary
