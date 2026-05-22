"""Quick check that the API key loads and the Anthropic client works."""
import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()  # reads .env into environment variables

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    print("ERROR: ANTHROPIC_API_KEY not found. Check your .env file.")
    raise SystemExit(1)

print("Key loaded OK. Length:", len(api_key))  # prints length, never the key

client = Anthropic(api_key=api_key)

resp = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=20,
    messages=[{"role": "user", "content": "Reply with exactly: API WORKS"}],
)
print("Model replied:", resp.content[0].text)