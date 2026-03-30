"""Local LLM loaded on GPU with 4-bit quantization."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import config


class LLM:
    def __init__(self):
        print(f"Loading LLM '{config.LLM_MODEL_NAME}' on {config.DEVICE}...")

        quantization_config = None
        if config.LLM_LOAD_IN_4BIT:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("LLM loaded.")

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=config.LLM_MAX_NEW_TOKENS,
                temperature=config.LLM_TEMPERATURE,
                top_p=config.LLM_TOP_P,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # Decode only the newly generated tokens
        generated_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
