import json

# Read the original file and skip the header
with open("train_chat.jsonl", "r") as f:
    lines = f.readlines()

# Write clean JSONL (skip first line which is the header)
with open("train_chat.jsonl", "w") as f:
    for line in lines[1:]:
        line = line.strip()
        if line:
            f.write(line + "\n")

print("JSONL file cleaned - removed header line")
