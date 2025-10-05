import json

try:
    with open("maestro-v3.0.0.json", encoding="utf-8") as f:
        data = json.load(f)
    print("JSON data successfully loaded:")
    print(f"Type of loaded data: {type(data)}")
except FileNotFoundError:
    print(f"Error: The file 'your_file.json' was not found.")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from 'your_file.json'. Check for valid JSON format.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


composers = list(data["canonical_composer"].values())
composers_count = {i:composers.count(i) for i in composers}
with open("composer_count.json", "w", encoding="utf-8") as f:  
    json.dump(composers_count, f, indent=4)