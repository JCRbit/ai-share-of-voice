import os
import json
import threading
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# Ensure .env is loaded before accessing os.getenv
load_dotenv()

GROQ_PRICING = {
    "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},
    "openai/gpt-oss-120b": {"input": 0.00015, "output": 0.00020},
}

class BudgetExceededError(Exception):
    pass

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float

class GroqClient:
    def __init__(self, spend_cap: float = 2.00):
        """
        Initializes the client per-run with a strict spend cap.
        """
        self.client = Groq(api_key=os.getenv("LLM_API_KEY"))
        self.model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
        
        if self.model not in GROQ_PRICING:
            raise ValueError(f"Unknown model configured: '{self.model}'. No pricing available.")
            
        self.rates = GROQ_PRICING[self.model]
        self.spend_cap = spend_cap
        self.current_spend = 0.0
        self._lock = threading.Lock()

    def complete(self, prompt: str, schema: type[BaseModel]) -> tuple[BaseModel, Usage]:
        # Explicitly inject the schema shape into the prompt
        schema_json = json.dumps(schema.model_json_schema())
        full_prompt = (
            f"{prompt}\n\n"
            f"You must respond with valid JSON matching this exact schema:\n{schema_json}"
        )
        
        # 1. Preflight Reservation (Concurrency-Safe)
        # Conservative estimate for input tokens (approx 1 token = 3 chars)
        est_input_tokens = len(full_prompt) // 3
        est_in_cost = (est_input_tokens / 1000.0) * self.rates["input"]
        
        with self._lock:
            available_budget = self.spend_cap - self.current_spend
            if est_in_cost >= available_budget:
                raise BudgetExceededError("Insufficient budget for estimated input tokens.")
                
            # Calculate maximum output tokens we can safely afford
            output_budget = available_budget - est_in_cost
            max_affordable_out_tokens = int((output_budget / self.rates["output"]) * 1000)
            
            # Constrain to reasonable model output limits to prevent massive reservations
            max_out_tokens = min(max_affordable_out_tokens, 4096)
            
            if max_out_tokens < 10:
                raise BudgetExceededError("Insufficient budget remaining for output tokens.")
                
            # Reserve the absolute maximum possible cost for this transaction
            max_possible_cost = est_in_cost + ((max_out_tokens / 1000.0) * self.rates["output"])
            self.current_spend += max_possible_cost

        # 2. Execute API Call
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                response_format={"type": "json_object"},
                max_tokens=max_out_tokens
            )
            
            content = response.choices[0].message.content
            result = schema.model_validate_json(content)
            
            pt = response.usage.prompt_tokens
            ct = response.usage.completion_tokens
            tt = response.usage.total_tokens
            actual_cost = (pt / 1000.0) * self.rates["input"] + (ct / 1000.0) * self.rates["output"]
            
            usage_stats = Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=tt, cost_usd=actual_cost)
            
        finally:
            # 3. Ledger Reconciliation
            with self._lock:
                self.current_spend -= max_possible_cost  # Refund the reserved maximum
                if 'actual_cost' in locals():
                    self.current_spend += actual_cost  # Apply the exact incurred cost
                else:
                    # If call failed before token usage was returned, apply estimated input cost
                    self.current_spend += est_in_cost

        return result, usage_stats
