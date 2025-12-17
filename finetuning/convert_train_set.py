import json

# 1. Load your raw data
with open('../teachers.json', 'r') as f:
    raw_data = json.load(f)

formatted_data = []

# 2. Iterate through records and transform them
for teacher in raw_data:
    # Create a natural language question (User)
    user_content = f"Tell me about {teacher['full_name']} from VVIT."

    # Create a natural language answer (Assistant)
    # We combine fields into a readable sentence or list
    assistant_content = (
        f"{teacher['full_name']} is a {teacher['designation']} in the "
        f"{teacher['department']} department at {teacher['institution']}. "
        f"Their highest qualification is {teacher['highest_qualification']}."
    )

    # 3. Structure into the "Messages" format (Best for Llama-3)
    entry = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for Vasireddy Venkatadri Institute of Technology (VVIT)."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }
    
    formatted_data.append(entry)

# 4. Save as JSONL (JSON Lines), which is the standard for Hugging Face
with open('train_chat.jsonl', 'w') as f:
    for entry in formatted_data:
        json.dump(entry, f)
        f.write('\n')

print(f"Successfully converted {len(raw_data)} records to 'train_chat.jsonl'")