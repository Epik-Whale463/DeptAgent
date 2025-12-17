import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Configuration
model_name = "unsloth/gemma-3-1b-it" # Standard 1B Instruct model
max_seq_length = 2048                # Gemma 3 supports up to 32K on 1B
load_in_4bit = True                  # Set to False if you want full BF16 training

# 2. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
    device_map = "auto",             # Automatically uses both L40S GPUs
)

# 3. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                          # Rank (Higher = more parameters, 16-64 is standard)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,                # Optimized at 0 for Unsloth
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
)

# 4. Load your local dataset
dataset = load_dataset("json", data_files="train_chat.jsonl", split="train")

# 5. Training Setup
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,               # Adjust based on your dataset size
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(), # L40S supports BF16
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 6. Start Training
print("Starting Fine-tuning on Dual L40S...")
trainer.train()

# 7. Export to GGUF for Ollama
# This creates a folder containing the GGUF file and Modelfile
print("Exporting to Ollama format...")
model.save_pretrained_gguf("vvit_gemma_model", tokenizer, quantization_method = "q4_k_m")

print("Fine-tuning complete. You can now use 'ollama create vvit_model -f vvit_gemma_model/Modelfile'")