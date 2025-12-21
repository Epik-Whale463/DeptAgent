from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# 1. Load Dataset
dataset = load_dataset("json", data_files="train_chat.jsonl", split="train")

# 2. Format for Mistral (using the chat template)
def format_instruction(sample):
    # Mistral v0.3 instruction format: <s>[INST] user_query [/INST] assistant_response </s>
    # Based on your train_chat.jsonl structure:
    # messages[0] = system, messages[1] = user, messages[2] = assistant
    user_query = sample['messages'][1]['content']
    assistant_resp = sample['messages'][2]['content']
    return f"<s>[INST] {user_query} [/INST] {assistant_resp} </s>"

# 3. Training Configuration (Replaces TrainingArguments + packing)
sft_config = SFTConfig(
    output_dir="./vvit-mistral-7b",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4, # Increased for better stability
    learning_rate=2e-4,
    num_train_epochs=5,           # Increased slightly for your small dataset
    bf16=True,                    # Optimized for L40S
    logging_steps=5,
    packing=False,                # Moved inside SFTConfig
    max_seq_length=512,           # Important to define this
    report_to="none",
)

# 4. Initialize Trainer
trainer = SFTTrainer(
    model=model_id,
    train_dataset=dataset,
    args=sft_config,
    formatting_func=format_instruction,
)

# 5. Start Training
trainer.train()

# 6. Save the model
trainer.save_model("./vvit-mistral-7b-final")