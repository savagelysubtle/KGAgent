"""Test the GPT chat parser."""
import sys
sys.path.insert(0, ".")

from src.kg_agent.parser.gpt_chat_parser import GPTChatParser

parser = GPTChatParser()

# Read the file
file_path = "data/raw/upload_20251125_004232_75d09ef4/31e06f7d89feb99a.html"
print(f"Testing file: {file_path}")

# Test full parsing
print("\n=== Testing full parse_file ===")
output_path = parser.parse_file(file_path, job_id="gpt_test")
if output_path:
    print(f"Output saved to: {output_path}")

    # Check the output
    import json
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    main_text = data.get("main_text", "")
    print(f"Main text length: {len(main_text):,} characters")
    print(f"Word count: ~{len(main_text.split()):,} words")
    print(f"Metadata: {data.get('_metadata', {})}")

    # Show first 1000 chars
    print("\n=== First 1000 characters of extracted text ===")
    print(main_text[:1000])
else:
    print("Parsing failed!")

