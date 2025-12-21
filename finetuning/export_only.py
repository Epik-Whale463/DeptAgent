from unsloth import FastLanguageModel

# 1. Load the model (Config only, no heavy weights needed)
model, tokenizer = FastLanguageModel.from_pretrained(
    "vvit_mistral_model",
    load_in_4bit = True,
)

# 2. Convert the EXISTING folder to GGUF
# We pass "vvit_mistral_model" because that is where the 16-bit weights already live.
print("Starting conversion on existing folder...")
model.save_pretrained_gguf(
    "vvit_mistral_model", # <--- Changed this to match your existing folder
    tokenizer,
    quantization_method = "q4_k_m"
)

print("\nSUCCESS! Run this command to import into Ollama:")
print("ollama create vvit_mistral -f vvit_mistral_model/Modelfile")