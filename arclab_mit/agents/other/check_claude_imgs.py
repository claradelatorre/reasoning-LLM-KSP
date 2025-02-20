import cv2
import base64
import anthropic
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read and encode the image
image_path = "C:/temp/debug_processed_image.jpg"
with open(image_path, "rb") as image_file:
    image_data = base64.b64encode(image_file.read()).decode('utf-8')

# Save a copy of the image in the script's directory
with open(os.path.join(script_dir, "last_captured_claude_image.jpg"), "wb") as f:
    f.write(base64.b64decode(image_data))

# Create the message
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What do you see in this image? This is from Kerbal Space Program. Can you describe the spacecraft position, prograde and relative velocity?"
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_data
                }
            }
        ]
    }
]

# Make the API call
response = client.messages.create(
    model="claude-3-sonnet-20240229",
    messages=messages,
    max_tokens=500
)

# Print the response
print("\nClaude's Response:")
if response.content:
    for content in response.content:
        print(content) 