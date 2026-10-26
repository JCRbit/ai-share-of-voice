import os
from pydantic import BaseModel, ValidationError
from groq import Groq
from ai_share_of_voice.exceptions import BudgetExceeded
from ai_share_of_voice.schemas.llm import Usage


class GroqClient:
    def __init__(self, spend_cap: float = 2.00):
        self.client = Groq(api_key=os.getenv("LLM_API_KEY"))
        self.model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        self.spend_cap = spend_cap
        self.current_spend = 0.0

        # Groq pricing (USD per token)
        self.cost_per_input_token = 0.59 / 1_000_000
        self.cost_per_output_token = 0.79 / 1_000_000

    def complete[T: BaseModel](self, prompt: str, schema: type[T]) -> tuple[T, Usage]:
        """Sends a prompt to Groq, tracks cost, and returns the requested Pydantic object and Usage stats."""
        if self.current_spend >= self.spend_cap:
            raise BudgetExceeded(spent_usd=self.current_spend, cap_usd=self.spend_cap)

        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You must output valid JSON matching the provided schema exactly.",
                },
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            response_format={"type": "json_object"},
        )

        api_usage = response.usage
        call_cost = (api_usage.prompt_tokens * self.cost_per_input_token) + (
            api_usage.completion_tokens * self.cost_per_output_token
        )
        self.current_spend += call_cost

        if self.current_spend > self.spend_cap:
            raise BudgetExceeded(spent_usd=self.current_spend, cap_usd=self.spend_cap)

        raw_json = response.choices[0].message.content
        try:
            parsed_data = schema.model_validate_json(raw_json)
        except ValidationError as e:
            raise ValueError(f"Groq returned invalid schema: {e}")

        usage_stats = Usage(
            model=self.model,
            input_tokens=api_usage.prompt_tokens,
            output_tokens=api_usage.completion_tokens,
            cost_usd=call_cost,
        )

        return parsed_data, usage_stats
