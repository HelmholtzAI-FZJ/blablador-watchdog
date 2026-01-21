from openai import OpenAI
from colorama import init, Fore, Style
from dotenv import load_dotenv
from embedding_models import EMBEDDING_MODELS
import os

# Load environment variables
load_dotenv()

# Initialize colorama
init(autoreset=True)

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

embedding_client = OpenAI(
    api_key=os.getenv("EMBEDDINGS_API_KEY") or os.getenv("API_KEY"),
    base_url=os.getenv("OPENAI_EMBEDDINGS_BASE_URL") or os.getenv("OPENAI_BASE_URL")
)



def get_available_models():
    try:
        models = client.models.list()
        return [model.id for model in models.data]
    except Exception as e:
        print(f"{Fore.RED}An error occurred while fetching models: {str(e)}")
        return []


def get_llm_response(prompt, model):
    response = None
    try:
        def request_completion(max_tokens):
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                top_p=1,
                n=1,
                max_tokens=max_tokens,
                stop=None,
                stream=False,
                presence_penalty=0,
                frequency_penalty=0,
                reasoning_effort="low",
            )

        response = request_completion(30)
        if response and hasattr(response, 'choices') and response.choices:
            message = getattr(response.choices[0], "message", None)
            content = getattr(message, "content", None) if message else None
            if isinstance(content, str):
                return content.strip()
            reasoning = None
            if message:
                reasoning = (
                    getattr(message, "reasoning_content", None)
                    or getattr(message, "reasoning", None)
                )
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            if reasoning and finish_reason == "length":
                response = request_completion(256)
                if response and hasattr(response, "choices") and response.choices:
                    message = getattr(response.choices[0], "message", None)
                    content = getattr(message, "content", None) if message else None
                    if isinstance(content, str):
                        return content.strip()
            print(f"{Fore.RED}Empty response content from model: {model}")
            return "An error occurred: Empty response content from LLM"
        else:
            print(f"{Fore.RED}Invalid response from model: {model}")
            return "An error occurred: Invalid response from LLM"
    except UnboundLocalError as e:
        print(f"{Fore.RED}Warning (UnboundLocalError) in {model}: {e}")
        # Continue execution: return empty string so caller can keep processing
        return ""
    except Exception as e:
        extra = ""
        if 'response' in locals() and response and hasattr(response, 'choices') and response.choices:
            try:
                message = getattr(response.choices[0], "message", None)
                content = getattr(message, "content", None) if message else None
                if isinstance(content, str):
                    extra = f" on {content}"
            except Exception:
                extra = ""
        print(f"{Fore.RED}Exception from model {model}: {e}{extra}")
        return f"An error occurred: {str(e)}{extra}"


def get_embedding_response(text, model):
    response = None
    try:
        response = embedding_client.embeddings.create(
            model=model,
            input=text,
        )
        if response and hasattr(response, "data") and response.data:
            embedding = getattr(response.data[0], "embedding", None)
            if isinstance(embedding, list) and embedding:
                return embedding
            print(f"{Fore.RED}Empty embedding response from model: {model}")
            return "An error occurred: Empty embedding response"
        else:
            print(f"{Fore.RED}Invalid embedding response from model: {model}")
            return "An error occurred: Invalid embedding response"
    except UnboundLocalError as e:
        print(f"{Fore.RED}Warning (UnboundLocalError) in {model}: {e}")
        return ""
    except Exception as e:
        print(f"{Fore.RED}Exception from model {model}: {e}")
        return f"An error occurred: {str(e)}"


# Main execution
if __name__ == "__main__":
    word = "potato"
    prompt = f"Give me ONLY a word. The word is {word}. Nothing else. \
        No sentences, no explanations, no definitions. Just the word."

    models = get_available_models()
    broken_models = []
    broken_gpus = []
    empty_response_models = []
    invalid_response_models = []
    empty_embedding_models = []
    invalid_embedding_models = []

    if not models:
        print(f"{Fore.RED}❌ No models available. Exiting.")
    else:
        print(f"{Fore.CYAN}🤖 Available models:")
        for model_name in models:
            print(f"{Fore.CYAN}  - {model_name}")
        for model in models:
            print(f"\n{Fore.YELLOW}📊 Testing model: {model}")
            if model in EMBEDDING_MODELS:
                print(f"{Fore.BLUE}🧠 Sending request to embedding model...")
                response = get_embedding_response(word, model)
                if response == "An error occurred: Empty embedding response":
                    empty_embedding_models.append(model)
                    broken_models.append(model)
                    print(f"{Fore.RED}❌ EMPTY EMBEDDING RESPONSE IN {model}: {response}")
                elif response == "An error occurred: Invalid embedding response":
                    invalid_embedding_models.append(model)
                    broken_models.append(model)
                    print(f"{Fore.RED}❌ INVALID EMBEDDING RESPONSE IN {model}: {response}")
                elif isinstance(response, list):
                    print(f"{Fore.GREEN}✅ Embedding length: {len(response)}")
                elif "CUDA error:" in response:
                    broken_models.append(model)
                    print(f"{Fore.RED}❌ HARDWARE FAILURE IN {model}: {response}")
                elif "Internal Server Error" in response:
                    broken_models.append(model)
                    print(f"{Fore.RED}❌ EMBEDDING FAILURE IN {model}: {response}")
                else:
                    broken_models.append(model)
                    print(f"{Fore.RED}❌ EMBEDDING response: {response}")
            else:
                print(f"{Fore.BLUE}🚀 Sending request to LLM...")
                response = get_llm_response(prompt, model)
                if response == "An error occurred: Empty response content from LLM":
                    empty_response_models.append(model)
                    broken_models.append(model)
                    print(f"{Fore.RED}❌ EMPTY LLM RESPONSE IN {model}: {response}")
                elif response == "An error occurred: Invalid response from LLM":
                    invalid_response_models.append(model)
                    broken_models.append(model)
                    print(f"{Fore.RED}❌ INVALID LLM RESPONSE IN {model}: {response}")
                elif word.lower() in response.lower():
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
        print(
            f"\n{Fore.RED}🔥🔥 Bad model response[s] ({len(broken_models)}): "
            f"{', '.join(broken_models)}"
        )

    if (
        empty_response_models
        or invalid_response_models
        or empty_embedding_models
        or invalid_embedding_models
    ):
        parts = []
        if empty_response_models:
            parts.append(
                f"empty({len(empty_response_models)}): {', '.join(empty_response_models)}"
            )
        if invalid_response_models:
            parts.append(
                f"invalid({len(invalid_response_models)}): {', '.join(invalid_response_models)}"
            )
        if empty_embedding_models:
            parts.append(
                f"embedding-empty({len(empty_embedding_models)}): {', '.join(empty_embedding_models)}"
            )
        if invalid_embedding_models:
            parts.append(
                f"embedding-invalid({len(invalid_embedding_models)}): {', '.join(invalid_embedding_models)}"
            )
        print(f"\n{Fore.RED}⚠️ Response issues: {'; '.join(parts)}")

    if broken_gpus:
        # **NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**\n\n(CUDA error: device-side assert triggered\nCUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.\nFor debugging consider passing CUDA_LAUNCH_BLOCKING=1\nCompile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.\n)', 'code': 50001}
        raise Exception(f"{Fore.RED}❌❌❌ These models should be restarted: {', '.join(broken_models)}")

    print(f"{Style.RESET_ALL}")  # Reset color at the end
