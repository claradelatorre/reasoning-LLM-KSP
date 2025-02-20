import cv2
import base64
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build path to prompt_example.jpg in the same directory
image_path = os.path.join(script_dir, "prompt_example.jpg")

# Read and encode the image
with open(image_path, "rb") as image_file:
    image_data = base64.b64encode(image_file.read()).decode('utf-8')

# Save to a file that matches our preamble path structure
output_path = os.path.join(script_dir, "..", "json_preambles", "vision_llm_agents", "prompt_example_base64.txt")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    f.write(image_data)

print(f"Base64 image data saved to: {output_path}") 