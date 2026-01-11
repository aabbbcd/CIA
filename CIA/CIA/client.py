import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_tokenizer = None
_model = None
_model_device = None


# def load_llm(model_path: str, device: str = "cuda:3", torch_dtype: str = "auto"):
# 	global _tokenizer, _model, _model_device
# 	if _model is not None:
# 		return _tokenizer, _model

# 	# 解析设备
# 	if device.startswith("cuda") and torch.cuda.is_available():
# 		# 支持 cuda 或 cuda:N
# 		_device = device
# 	else:
# 		_device = "cpu"

# 	# dtype
# 	if torch_dtype == "auto":
# 		dtype = torch.float16 if _device.startswith("cuda") else torch.float32
# 	elif torch_dtype == "float16":
# 		dtype = torch.float16
# 	elif torch_dtype == "bfloat16":
# 		dtype = torch.bfloat16
# 	else:
# 		dtype = torch.float32

# 	_tokenizer = AutoTokenizer.from_pretrained(model_path)
# 	_model = AutoModelForCausalLM.from_pretrained(
# 		model_path,
# 		torch_dtype=dtype,
# 		low_cpu_mem_usage=True,
# 		device_map="auto" if _device.startswith("cuda") else None
# 	)
# 	if not _device.startswith("cuda"):
# 		_model = _model.to(_device)
# 	_model.eval()
# 	_model_device = _device
# 	return _tokenizer, _model


# def generate(prompt: str,
# 			 model_path: str,
# 			 device: str = "cuda:3",
# 			 max_new_tokens: int = 512,
# 			 temperature: float = 0.01,
# 			 do_sample: bool = False) -> str:
# 	"""生成文本（不加载多次模型，走单例缓存）。"""
# 	tok, mdl = load_llm(model_path=model_path, device=device)
# 	inputs = tok(prompt, return_tensors="pt")
# 	inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
# 	with torch.no_grad():
# 		out = mdl.generate(
# 			**inputs,
# 			max_new_tokens=max_new_tokens,
# 			temperature=max(1e-3, float(temperature)),
# 			do_sample=do_sample,
# 			pad_token_id=tok.eos_token_id,
# 			eos_token_id=tok.eos_token_id
# 		)
# 	text = tok.decode(out[0], skip_special_tokens=True)
# 	return text

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
				return f"生成失败: {e}"
		prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
		result = generate_text_transformers(prompt)
		return result