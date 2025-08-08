from openai import OpenAI
from colorama import init, Fore, Style
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize colorama
init(autoreset=True)

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
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
            frequency_penalty=0,
            reasoning_effort="low",
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {str(e)} on {response.choices[0].message.content}"


# Main execution
if __name__ == "__main__":
    word = "potato"
    prompt = f"Give me ONLY a word. The word is {word}. Nothing else. \
        No sentences, no explanations, no definitions. Just the word."

    models = get_available_models()
    broken_models = []
    broken_gpus = []

    if not models:
        print(f"{Fore.RED}❌ No models available. Exiting.")
    else:
        print(f"{Fore.CYAN}🤖 Available models:")
        for model_name in models:
            print(f"{Fore.CYAN}  - {model_name}")
        for model in models:
            print(f"\n{Fore.YELLOW}📊 Testing model: {model}")
            print(f"{Fore.BLUE}🚀 Sending request to LLM...")
            response = get_llm_response(prompt, model)
            if word.lower() in response.lower():
                print(f"{Fore.GREEN}✅ LLM response: {response}")
            elif "CUDA error:" in response:
                broken_models.append(model)
                print(f"{Fore.RED}❌ HARDWARE FAILURE IN {model}: {response}")
            elif "Internal Server Error" in response:
                broken_models.append(model)
                print(f"{Fore.RED}❌ LLM FAILURE IN {model}: {response}")
            else:
                broken_models.append(model)
                print(f"{Fore.RED}❌ LLM response: {response}")

    if broken_models:
        print(f"\n{Fore.RED}🔥🔥 Bad model response[s]: {', '.join(broken_models)}")

    if broken_gpus:
        # **NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**\n\n(CUDA error: device-side assert triggered\nCUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.\nFor debugging consider passing CUDA_LAUNCH_BLOCKING=1\nCompile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.\n)', 'code': 50001}
        raise Exception(f"{Fore.RED}❌❌❌ These models should be restarted: {', '.join(broken_models)}")

    print(f"{Style.RESET_ALL}")  # Reset color at the end
