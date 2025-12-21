from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# 1. Load Dataset
dataset = load_dataset("json", data_files="train_chat.jsonl", split="train")

# 2. Format for Mistral
def format_instruction(sample):
    user_query = sample['messages'][1]['content']
    assistant_resp = sample['messages'][2]['content']
    return f"<s>[INST] {user_query} [/INST] {assistant_resp} </s>"

# 3. Training Configuration (Remove max_seq_length from here)
sft_config = SFTConfig(
    output_dir="./vvit-mistral-7b",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=5,
    bf16=True,
    logging_steps=5,
    packing=False,
    report_to="none",
)

# 4. Initialize Trainer (Pass max_seq_length here instead)
trainer = SFTTrainer(
    model=model_id,
    train_dataset=dataset,
    args=sft_config,
    formatting_func=format_instruction,
    max_seq_length=512, # Pass as a direct argument
)

# 5. Start Training
trainer.train()
trainer.save_model("./vvit-mistral-7b-final")