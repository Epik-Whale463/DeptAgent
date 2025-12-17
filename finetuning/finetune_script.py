import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template # ADD THIS

# 1. Configuration
model_name = "unsloth/gemma-3-1b-it" 
max_seq_length = 2048                
load_in_4bit = True                  

# 2. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
    device_map = "auto",             
)

# 3. Add Template & Formatting Function (THE FIX)
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma", # Use Gemma specific tokens
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

# 4. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                          
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,                
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
)

# 5. Load and Map Dataset
dataset = load_dataset("json", data_files="train_chat.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,) # ADD THIS MAP STEP

# 6. Training Setup
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text", # Now matches the output of formatting_prompts_func
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,               
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(), 
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 7. Start Training
print("Starting Fine-tuning on Dual L40S...")
trainer.train()

# 8. Export to GGUF for Ollama
print("Exporting to Ollama format...")
model.save_pretrained_gguf("vvit_gemma_model", tokenizer, quantization_method = "q4_k_m")

print("Complete! Use: 'ollama create vvit_model -f vvit_gemma_model/Modelfile'")