import os

import pytest
from pydantic import BaseModel

from llmcall import extract, generate, generate_decision

api_key = os.getenv("LLMCALL_API_KEY")
pytestmark = pytest.mark.skipif(
    not api_key,
    reason="LLMCALL_API_KEY not configured",
)


class CountrySchema(BaseModel):
    name: str
    capital: str
    population_millions: float


def test_live_generate():
    response = generate(
        prompt="Tell me a joke about a developer and empty lists",
        instructions="Return only the joke text.",
    )
    assert isinstance(response, str)
    assert len(response) > 0


def test_live_generate_with_schema():
    response = generate(
        prompt="What is the capital of France and its estimated population?",
        output_schema=CountrySchema,
    )
    assert isinstance(response, CountrySchema)
    assert "france" in response.name.lower() or response.name == "France"
    assert "paris" in response.capital.lower() or response.capital == "Paris"
    assert response.population_millions > 0


def test_live_generate_decision():
    prompt = (
        "I need a database system to query structured server metrics "
        "highly performantly with SQL-like syntax. Which engine is better?"
    )
    decision = generate_decision(
        prompt=prompt,
        options=["PostgreSQL", "InfluxDB", "ClickHouse"],
    )
    assert decision.selection in ["PostgreSQL", "InfluxDB", "ClickHouse"]
    assert (
        decision.selection == "ClickHouse"
        or decision.selection == "InfluxDB"
        or decision.selection == "PostgreSQL"
    )


def test_live_extract():
    text = (
        "Alice works as a Software Engineer at Acme Corp. "
        "She recently negotiated a salary of 120000 USD per year."
    )

    class EmploymentDetails(BaseModel):
        employee_name: str
        role: str
        company: str
        salary_usd: int

    result = extract(text=text, output_schema=EmploymentDetails)
    assert isinstance(result, EmploymentDetails)
    assert result.employee_name == "Alice"
    assert "engineer" in result.role.lower()
    assert result.company == "Acme Corp"
    assert result.salary_usd == 120000
