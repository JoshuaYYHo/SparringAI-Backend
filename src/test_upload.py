import requests
import json
import os
import sys
import mimetypes
from dotenv import load_dotenv

# Load the .env file to get the SPARRING_API_KEY automatically
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

def test_upload(video_path):
    url = "http://localhost:8000/video_upload"
    api_key = os.getenv("SPARRING_API_KEY")
    
    if not api_key:
        print("Error: SPARRING_API_KEY is not set in your .env file!")
        return
        
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    headers = {
        "x-api-key": api_key
    }

    params = {
        "confidence_threshold": 0.7,
        # "user_bbox": "100,200,300,400"  # Uncomment if you know your bounding box
    }

    print(f"Uploading '{video_path}' to {url}...")
    print("This might take a few minutes as the server processes the video and OpenAI generates the summary.")

    try:
        # Determine MIME type based on extension
        mime_type, _ = mimetypes.guess_type(video_path)
        if not mime_type:
            mime_type = "video/mp4" # Fallback
            
        with open(video_path, "rb") as video_file:
            # Format: ('filename', <file object>, 'content-type')
            files = {"video": (os.path.basename(video_path), video_file, mime_type)}
            response = requests.post(url, headers=headers, params=params, files=files)
    except requests.exceptions.ConnectionError:
        print("Error: Couldn't connect to the server. Make sure your Docker container is running properly on port 8000.")
        return

    if response.status_code == 200:
        data = response.json()
        print("\n✅ SUCCESS!")
        print("\n================ AI Fight Summary ================\n")
        print(data.get("ai_commentary", "No commentary returned"))
        print("\n==================================================\n")
    else:
        print(f"\n❌ Server returned an error (Status {response.status_code}):")
        print(response.text)

if __name__ == "__main__":
    # Change this path to whichever video you want to test!
    VIDEO_PATH = "example_videos/example_3.mp4"
    
    test_upload(VIDEO_PATH)
