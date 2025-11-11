import json
import csv
from collections import Counter

# load MAESTRO JSON data
with open("metadata/maestro-v3.0.0.json", encoding="utf-8") as f:
    data = json.load(f)

# count occurrences of each composer
composer_count = Counter(data['canonical_composer'].values())

def json_to_csv(json_data, csv_filename, headers):
    """Convert a JSON file to a CSV file."""
    
    with open(csv_filename, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(json_data.items())

ordered_composer_count = dict(sorted(composer_count.items(), key=lambda item: item[1], reverse=True))
json_to_csv(ordered_composer_count, 'metadata/composer_count.csv', ['Composer', 'Count'])

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

columnar_json_to_csv(data, 'metadata/maestro-v3.0.0.csv')