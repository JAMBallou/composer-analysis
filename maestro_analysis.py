import json
import csv

# load MAESTRO JSON data
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



# count occurrences of each composer
composers = list(data["canonical_composer"].values())
composers_count = {i:composers.count(i) for i in composers}

def json_to_csv(json_data, csv_filename, headers):
    """Convert a JSON file to a CSV file."""
    
    with open(csv_filename, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        
        writer.writerow(headers)
        writer.writerows(json_data.items())

# sort composer_count by number of works
sorted_composer_count = dict(sorted(composers_count.items(), key=lambda item: item[1], reverse=True))
json_to_csv(sorted_composer_count, 'composer_count.csv', ['Composer', 'Count'])



def columnar_json_to_csv(json_data, csv_filename):
    # fetch indices
    all_indices = set()
    for column in json_data.values():
        all_indices.update(column.keys())
    # ensure sorted order
    all_indices = sorted(all_indices, key=int)
    
    # fetch column headers
    headers = list(json_data.keys())

    # to CSV
    with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Index'] + headers)
        
        for idx in all_indices:
            row = [idx]
            for head in headers:
                value = json_data.get(head, {}).get(idx, "")
                row.append(value)
            writer.writerow(row)   

columnar_json_to_csv(data, 'maestro-v3.0.0.csv')