import os

from groq import Groq
from pydantic import BaseModel, ValidationError

from ai_share_of_voice.exceptions import BudgetExceeded
from ai_share_of_voice.schemas.llm import Usage

GROQ_PRICING = {
    "llama-3.3-70b-versatile": {"input": 0.59 / 1_000_000, "output": 0.79 / 1_000_000},
    "llama3-70b-8192": {"input": 0.59 / 1_000_000, "output": 0.79 / 1_000_000},
    "llama3-8b-8192": {"input": 0.05 / 1_000_000, "output": 0.08 / 1_000_000},
    "mixtral-8x7b-32768": {"input": 0.24 / 1_000_000, "output": 0.24 / 1_000_000},
    "default": {"input": 0.59 / 1_000_000, "output": 0.79 / 1_000_000},
}


class GroqClient:
    def __init__(self, spend_cap: float = 2.00):
        self.client = Groq(api_key=os.getenv("LLM_API_KEY"))
        self.model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        self.spend_cap = spend_cap
        self.current_spend = 0.0

        # Dynamic Groq pricing based on configured model
        rates = GROQ_PRICING.get(self.model, GROQ_PRICING["default"])
        self.cost_per_input_token = rates["input"]
        self.cost_per_output_token = rates["output"]

    def complete[T: BaseModel](self, prompt: str, schema: type[T]) -> tuple[T, Usage]:
        """
        Sends a prompt to Groq, tracks cost, and returns the requested
        Pydantic object and Usage stats.
        """
        # 1. Preflight check: Estimate input cost (approx 1 token = 4 chars)
        estimated_input_tokens = len(prompt) // 4
        estimated_input_cost = estimated_input_tokens * self.cost_per_input_token

        if self.current_spend + estimated_input_cost >= self.spend_cap:
            raise BudgetExceeded(spent_usd=self.current_spend, cap_usd=self.spend_cap)

        # 2. Preflight check: Constrain output tokens based on remaining budget
        remaining_budget = self.spend_cap - (self.current_spend + estimated_input_cost)
        calculated_max = int(remaining_budget / self.cost_per_output_token)

        # API constraint: max_tokens cannot exceed model limits (using 8192 as a safe default)
        max_safe_output_tokens = min(calculated_max, 8192)

        if max_safe_output_tokens < 50:
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
            max_tokens=max_safe_output_tokens,
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
            raise ValueError(f"Groq returned invalid schema: {e}") from e

        usage_stats = Usage(
            model=self.model,
            input_tokens=api_usage.prompt_tokens,
            output_tokens=api_usage.completion_tokens,
            cost_usd=call_cost,
        )

        return parsed_data, usage_stats
