import google.generativeai as genai

# Your specific key
API_KEY = "AIzaSyBgtlYIYFjyqZlLkFtB0QK0oTTAVIUdauo"

print(f"? Checking models for key ending in ...{API_KEY[-4:]}")

try:
    genai.configure(api_key=API_KEY)
    print("\n? AVAILABLE MODELS:")
    print("-------------------")

    found = False
    for m in genai.list_models():
        # We only care about models that can generate text (Chat)
        if 'generateContent' in m.supported_generation_methods:
            print(f" - {m.name}")
            found = True

    if not found:
        print("?? No chat models found. Check your API Key permissions.")

except Exception as e:
    print(f"\n? ERROR: {e}")
    print("Your API Key might be invalid or you have no internet.")
