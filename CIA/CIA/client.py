import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_tokenizer = None
_model = None
_model_device = None




def generate(messages: str,
			 model_path: str,
			 device: str = "cuda:0",
			 max_new_tokens: int = 512,
			 temperature: float = 0.01,
			 do_sample: bool = False) -> str:
		global _tokenizer, _model
		device_map = device 
		if _model is None:
			_tokenizer = AutoTokenizer.from_pretrained(model_path)
			_model = AutoModelForCausalLM.from_pretrained(
				model_path,
				torch_dtype=torch.float16,  
				device_map=device_map,		  
				low_cpu_mem_usage=True
			)
		tokenizer = _tokenizer
		model = _model


	
		def generate_text_transformers(prompt, max_new_tokens=10000):
			try:
				model.eval()
				inputs = tokenizer(prompt, return_tensors="pt")
				device = getattr(model, "device", next(model.parameters()).device)
				inputs = {k: v.to(device) for k, v in inputs.items()}

				with torch.no_grad():
					outputs = model.generate(
						**inputs,
						do_sample=True,
						temperature=0.5,
						top_p=0.5,
						max_new_tokens=max_new_tokens,
						pad_token_id=tokenizer.eos_token_id,
						eos_token_id=getattr(tokenizer, "eos_token_id", None),
					)

				prompt_len = inputs["input_ids"].shape[1]
				gen_ids = outputs[:, prompt_len:]
				text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
				return text

			except Exception as e:
				return f"error: {e}"
		prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
		result = generate_text_transformers(prompt)
		return result
