"""
JSON to JSONL Converter
=======================

Converts simulator_output.json to vitals.jsonl format expected by dataset generator.
"""

import json
import os

def convert_json_to_jsonl(input_file, output_file):
    """
    Convert JSON array to JSONL format.
    
    Args:
        input_file: path to JSON file
        output_file: path to output JSONL file
    """
    print(f"Converting {input_file} to {output_file}...")
    
    # Read JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Check if it's a list
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and 'records' in data:
        records = data['records']
    else:
        print("Error: Unexpected JSON format")
        return
    
    # Write JSONL
    with open(output_file, 'w') as f:
        for record in records:
            json.dump(record, f)
            f.write('\n')
    
    print(f"Converted {len(records)} records successfully!")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    # Default paths
    input_file = "../../data/simulator_output.json"
    output_file = "../../data/vitals.jsonl"
    
    # Check if input exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        print("\nPlease ensure simulator_output.json exists in ../data/ directory")
        exit(1)
    
    # Convert
    convert_json_to_jsonl(input_file, output_file)
    
    print("\n✓ Conversion complete! You can now run generate_dataset.py")
