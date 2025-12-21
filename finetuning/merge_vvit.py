import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_path = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_path = "./vvit-mistral-7b-final"
save_path = "./vvit-mistral-7b-merged"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu" # Use CPU to save GPU memory for the merge process
)

print("Merging weights...")
model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = model.merge_and_unload() # Permanently combines LoRA and Base

print("Saving merged model...")
merged_model.save_pretrained(save_path)
tokenizer = AutoTokenizer.from_pretrained(base_path)
tokenizer.save_pretrained(save_path)