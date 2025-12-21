import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Define paths
base_model_path = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_path = "./vvit-mistral-7b-final"

# 2. Load the base model (using bf16 to match your L40S capability)
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

# 4. Define the test prompt
# Use the same Mistral [INST] format used during training
question = "Who is the Chairman of VVIT and what is the college's NAAC grade?"
prompt = f"<s>[INST] {question} [/INST]"

# 5. Generate Response
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=150, 
        temperature=0.7, 
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# 6. Print Output
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n" + "="*30)
print(f"Query: {question}")
print(f"Response: {response.split('[/INST]')[-1].strip()}")
print("="*30)