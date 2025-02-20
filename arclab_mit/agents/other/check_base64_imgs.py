import cv2
import base64
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read and encode the image
image_path = "C:/temp/debug_processed_image.jpg"  # From project root
with open(image_path, "rb") as image_file:
    image_data = base64.b64encode(image_file.read()).decode('utf-8')

# Save a copy of the image in the script's directory
with open(os.path.join(script_dir, "last_captured_image.jpg"), "wb") as f:
    f.write(base64.b64decode(image_data))

# Create the message
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What do you see in this image? This is from Kerbal Space Program. Can you say if you are getting closer or further? Also, how could you transfer to the other spacecraft's orbit?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}",
                    "detail": "high"
                }
            }
        ]
    }
]

# Make the API call
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    max_tokens=500
)

# Print the response
print(response.choices[0].message.content)