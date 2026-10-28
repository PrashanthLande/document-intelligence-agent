from typing import Protocol

from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam


class Generator(Protocol):
    """An LLM backend: turns a prompt into a reply."""

    def generate(self, prompt: str) -> str: ...


class TransformersGenerator:
    """Run a Hugging Face chat model in-process with transformers."""

    def __init__(
            self,
            model_name: str = "Qwen/Qwen3-1.7B",
            max_new_tokens: int = 512,
    ) -> None:
        # Imported here so remote backends don't need torch loaded.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
        ).to(device)
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            # Qwen3 only; other chat templates ignore it.
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
        )

        # The output repeats the prompt; keep only the new tokens.
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        return self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()


class OpenAICompatibleGenerator:
    """Call any server that speaks the OpenAI Chat Completions API.

    Covers OpenAI itself, Groq, vLLM, Ollama and llama.cpp server.
    """

    def __init__(
            self,
            model_name: str,
            base_url: str | None = None,
            api_key: str | None = None,
    ) -> None:
        # Local servers usually need no key, but the client requires one.
        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-set")
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": prompt,
        }

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[message],
        )

        return response.choices[0].message.content or ""
