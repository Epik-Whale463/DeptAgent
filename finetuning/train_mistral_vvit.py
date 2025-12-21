from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# 1. Load Dataset
dataset = load_dataset("json", data_files="train_chat.jsonl", split="train")

# 2. Format for Mistral
def format_instruction(sample):
    # Mistral Instruction format
    return f"<s>[INST] {sample['messages'][1]['content']} [/INST] {sample['messages'][2]['content']} </s>"

# 3. Training Arguments
training_args = TrainingArguments(
    output_dir="./vvit-mistral-7b",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=3,
    bf16=True, # L40S supports bf16
    logging_steps=10,
    push_to_hub=False,
    report_to="none"
)

# 4. Initialize Trainer
trainer = SFTTrainer(
    model=model_id,
    train_dataset=dataset,
    packing=False,
    formatting_func=format_instruction,
    args=training_args,
)

trainer.train()