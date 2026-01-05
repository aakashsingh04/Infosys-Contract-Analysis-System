
from free_llm_provider import FreeLLMProvider
import sys

print("Testing FreeLLMProvider...")
try:
    llm = FreeLLMProvider.get_free_llm()
    if llm:
        print("Success! Got LLM.")
    else:
        print("No LLM found (expected if no keys and no Ollama).")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
