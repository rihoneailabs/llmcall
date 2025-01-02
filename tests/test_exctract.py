from typing import Optional
from llmcall import extract
from pydantic import BaseModel


def test_extract_from_unstructured_text():
    text = "We are looking for qualified Uber Drivers in Cape Town. If you are interested, please contact us at 021 123 4567. Salary is R5000 per month (negotiable)."

    class JobAd(BaseModel):
        title: str
        location: str
        contact: Optional[str]
        salary: Optional[str]

    result = extract(text=text, output_schema=JobAd)
    assert result.title.lower() == "uber driver"
    assert result.location.lower() == "cape town"
    assert result.contact == "021 123 4567"
    assert "R5000" in result.salary
