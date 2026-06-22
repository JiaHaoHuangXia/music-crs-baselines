from .llama import LLAMA_MODEL

class TEMPLATE_MODEL:
    """Lightweight response generator for retrieval-focused experiments."""

    def response_generation(self, sys_prompt: str, chat_history: list, recommend_item: str, max_new_tokens=512, response_format=None):
        return f"I think this track fits your request: {recommend_item}"

    def batch_response_generation(self, sys_prompts: list[str], chat_histories: list[list], recommend_items: list[str], max_new_tokens=128):
        return [
            f"I think this track fits your request: {recommend_item}"
            for recommend_item in recommend_items
        ]

def load_lm_module(lm_type, device, attn_implementation, dtype):
    if lm_type == "meta-llama/Llama-3.2-1B-Instruct":
        return LLAMA_MODEL(model_name=lm_type, device=device, attn_implementation=attn_implementation, dtype=dtype)
    elif lm_type == "template":
        return TEMPLATE_MODEL()
    else:
        raise ValueError(f"Unsupported LM type: {lm_type}")
