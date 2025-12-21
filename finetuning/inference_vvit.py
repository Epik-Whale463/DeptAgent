import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Define paths
base_model_path = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_path = "./vvit-mistral-7b-final"

# 2. Load the base model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 3. Load the LoRA adapter
print("Loading VVIT adapters...")
model = PeftModel.from_pretrained(base_model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# --- CRITICAL FIXES START ---
model.eval() # 1. Set to evaluation mode to stop gradient/dropout behavior
tokenizer.pad_token = tokenizer.eos_token # 2. Fix uninitialized padding token
tokenizer.padding_side = "left" # 3. Causal LMs must pad on the left for inference
# --- CRITICAL FIXES END ---

# 4. Define the test prompt
question = "Who is the Chairman of VVIT and what is the college's NAAC grade?"
prompt = f"<s>[INST] {question} [/INST]"

# 5. Generate Response with stability settings
inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")

print("Generating...")
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=150,
        do_sample=False, # 4. Use Greedy decoding first; it is mathematically safer
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

# 6. Print Output
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n" + "="*30)
print(f"Query: {question}")
print(f"Response: {response.split('[/INST]')[-1].strip()}")
print("="*30)