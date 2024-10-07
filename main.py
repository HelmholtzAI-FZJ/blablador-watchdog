from openai import OpenAI
from colorama import init, Fore, Style
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize colorama
init(autoreset=True)

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://helmholtz-blablador.fz-juelich.de:8000/v1"
)

def get_available_models():
    try:
        models = client.models.list()
        return [model.id for model in models.data]
    except Exception as e:
        print(f"{Fore.RED}An error occurred while fetching models: {str(e)}")
        return []

def get_llm_response(prompt, model):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            top_p=1,
            n=1,
            max_tokens=30,
            stop=None,
            stream=False,
            presence_penalty=0,
            frequency_penalty=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Main execution
if __name__ == "__main__":
    word = "potato"
    prompt = f"Give me ONLY a word. The word is {word}. Nothing else. No sentences, no explanations, no definitions. Just the word."
    
    models = get_available_models()
    
    if not models:
        print(f"{Fore.RED}❌ No models available. Exiting.")
    else:
        print(f"{Fore.CYAN}🤖 Available models: {', '.join(models)}")
        
        for model in models:
            print(f"\n{Fore.YELLOW}📊 Testing model: {model}")
            print(f"{Fore.BLUE}🚀 Sending request to LLM...")
            response = get_llm_response(prompt, model)
            if word.lower() in response.lower():
                print(f"{Fore.GREEN}✅ LLM response: {response}")
            else:
                print(f"{Fore.RED}❌ LLM response does not contain the word '{word}'. Reply: {response}")

    print(f"{Style.RESET_ALL}")  # Reset color at the end