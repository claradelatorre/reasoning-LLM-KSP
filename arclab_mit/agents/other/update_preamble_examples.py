import json
import os
import base64

def update_preamble_with_examples():
    # Get the directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    preamble_dir = os.path.join(script_dir, "..", "json_preambles", "vision_llm_agents")
    examples_dir = os.path.join(preamble_dir, "examples")
    
    # Ensure examples directory exists
    os.makedirs(examples_dir, exist_ok=True)
    
    # Load the preamble
    preamble_path = os.path.join(preamble_dir, "gpt_4o-2024-05-13_preamble.json")
    with open(preamble_path, 'r') as f:
        preamble = json.load(f)
    
    # Debug print
    # print("Preamble structure:", json.dumps(preamble, indent=2))
    
    # Update each example in sequence
    example_count = 1
    for i, message in enumerate(preamble["messages"]):
        # print(f"\nProcessing message {i}:", json.dumps(message, indent=2))
        
        if message["role"] == "user" and isinstance(message["content"], list):
            example_path = os.path.join(examples_dir, f"example_{example_count}.jpg")
            # print(f"Looking for example at: {example_path}")
            
            if os.path.exists(example_path):
                with open(example_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                
                # Find the image_url in the content array
                for j, content_item in enumerate(message["content"]):
                    # print(f"Checking content item {j}:", json.dumps(content_item, indent=2))
                    
                    if isinstance(content_item, dict) and content_item.get("type") == "image_url":
                        content_item["image_url"]["url"] = f"data:image/jpeg;base64,{image_data}"
                        print(f"✓ Updated preamble with example_{example_count}.jpg")
                        example_count += 1
                        break
            else:
                print(f"✗ Example image not found at: {example_path}")
                return None
    
    return preamble

if __name__ == "__main__":
    updated_preamble = update_preamble_with_examples()
    if updated_preamble:
        print("\nSuccessfully updated preamble")
    else:
        print("\nFailed to update preamble") 