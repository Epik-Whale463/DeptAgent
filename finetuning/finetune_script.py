import os
# FORCE Single GPU mode
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Better error messages

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template

# 1. Configuration
model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
max_seq_length = 2048
load_in_4bit = True

# 2. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
    device_map = "auto",
)

# 3. Apply the Mistral Chat Template (NOT Llama-3.2 for Mistral models)
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "mistral", # Use "mistral" for Mistral models
)

# 4. No custom formatting needed - dataset is in chat format with "messages"

def formatting_func(examples):
    """Convert messages to formatted text using the chat template"""
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

# 5. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 6. Load and Format Dataset
# Chat format JSONL - dataset already has "messages" field
dataset = load_dataset("json", data_files="train_chat.jsonl", split="train")
dataset = dataset.map(formatting_func, batched=True)

# 7. Training Arguments
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,  # Disable packing to avoid tokenization issues
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 1,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = False,
        bf16 = True,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 8. Train
print("Starting Stable Mistral Training...")
trainer.train()

# 9. Export to Ollama (GGUF)
print("Exporting to Ollama format...")
model.save_pretrained_gguf("vvit_mistral_model", tokenizer, quantization_method = "q4_k_m")

print("Done! Run: 'ollama create vvit_mistral -f vvit_mistral_model/Modelfile'")