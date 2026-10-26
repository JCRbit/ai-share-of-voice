import os
from dotenv import load_dotenv
from ai_share_of_voice.llm_client import GroqClient
from ai_share_of_voice.schemas.analysis import AnalyzedResponse

load_dotenv()

def run_test():
    cap = float(os.getenv("SPEND_CAP_USD", 2.00))
    client = GroqClient(spend_cap=cap)
    
    print(f"🚀 Testing Groq API... (Cap: ${cap})")
    
    prompt = """
    Analyze this text for mentions of the brand 'Nike'. 
    Text: 'I really love my new Nike running shoes, they are highly recommended for marathons. Adidas is okay but not my favorite.'
    
    You must return a JSON object with the following structure exactly:
    {
        "response_id": "test-123",
        "question_id": "q-1",
        "batch_index": 0,
        "mentions": [
            {
                "brand": "Nike",
                "mentioned": true,
                "recommended": true,
                "position": 1,
                "tone": "positive"
            }
        ],
        "sources": ["https://example.com/test"]
    }
    """
    
    try:
        # Pass the prompt and the Pydantic schema class
        result, usage = client.complete(prompt, AnalyzedResponse)
        print("\n✅ Success! Valid Pydantic object received:")
        print(result.model_dump_json(indent=2))
        print(f"\n📊 Usage Stats: {usage.total_tokens} total tokens")
        print(f"💰 Current Spend: ${client.current_spend:.6f}")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    run_test()