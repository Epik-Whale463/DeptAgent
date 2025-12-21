from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# 1. Load Dataset
dataset = load_dataset("json", data_files="train_chat.jsonl", split="train")

# 2. Format for Mistral
def format_instruction(sample):
    user_query = sample['messages'][1]['content']
    assistant_resp = sample['messages'][2]['content']
    return f"<s>[INST] {user_query} [/INST] {assistant_resp} </s>"

# 3. LoRA Configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 4. Training Configuration
sft_config = SFTConfig(
    output_dir="./vvit-mistral-7b",
    per_device_train_batch_size=4, # Increased back to 4 for L40S
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=5,
    bf16=True,
    logging_steps=5,
    packing=False,
    max_length=512,
    gradient_checkpointing=False, # DISABLED to fix the DDP error
    report_to="none",
)

# 5. Initialize Model & Tokenizer
# Explicitly setting device_map to ensure DDP handles placement
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16,
    device_map=None # Let Accelerator/DDP handle this
)

# 6. Initialize Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    formatting_func=format_instruction,
    peft_config=lora_config,
)

# 7. Start Training
trainer.train()
trainer.save_model("./vvit-mistral-7b-final")