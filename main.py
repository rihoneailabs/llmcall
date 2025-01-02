import json
from llmcall import generate

from pydantic import BaseModel

prompt = "What is the capital of France?"
response = generate(prompt)
print(response)
class ResponseFormat(BaseModel):
    name: str
response = generate(prompt, output_schema=ResponseFormat)
print(type(response))
print(vars(response))
out = ResponseFormat.model_validate(json.loads(response.choices[0].message.content))
print(out.name)