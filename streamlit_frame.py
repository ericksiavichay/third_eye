import streamlit as st
import requests
from PIL import Image
import io
from elevenlabs import set_api_key
from io import BytesIO

# Main application
def main():
    st.title("thirdeye")
    set_api_key("1da2bb9407e638f4445d437c8e8770e1")

    # Widget: Capture image from user's webcam
    # img_file_buffer = st.video("/")
    img_file_buffer = '/home/bala/third_eye/6.jpg'
    # Display image provided by user
    
    st.image(img_file_buffer)
    
    image = Image.open(img_file_buffer)
    imgByteArr = io.BytesIO()
  # image.save expects a file-like as a argument
    image.save(imgByteArr, format=image.format)
    # Turn the BytesIO object back into a bytes object
    imgByteArr = imgByteArr.getvalue()
    # Send the image to your API and get the description back.
    image_files = {"file": ("image.jpg", imgByteArr, "image/jpeg")}
    error_message = "Sorry, didnt work"
    description = error_message

    try:
        image_response = requests.post(
            "http://biksjzk4nu.loclx.io/uploadfile", files=image_files
        )
        image_response.raise_for_status()
        response_json = image_response.json()

        # Extract description from API response
        description = response_json.get(
            "text", "No description provided by the API"
        )
    except requests.RequestException as e:
        st.warning("Failed to communicate with the description API.")
        st.error(e)

    # Display the description
    st.subheader("Description from API:")
    st.write(description)

    # Send the description to ElevenLabs API for text-to-speech synthesis.
    tts_url = "https://api.elevenlabs.io/v1/text-to-speech/D38z5RcWu1voky8WS1ja"
    tts_headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": "1da2bb9407e638f4445d437c8e8770e1",
    }
    tts_data = {
        "text": description,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
    }

    try:
        tts_response = requests.post(tts_url, json=tts_data, headers=tts_headers)
        tts_response.raise_for_status()

        # Save audio file and provide playback
        audio_file = "output.mp3"
        with open(audio_file, "wb") as f:
            for chunk in tts_response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        audio_file_path = audio_file
        st.audio(audio_file_path, format="audio/mp3")
    except requests.RequestException as e:
        st.warning("Failed to communicate with the ElevenLabs API.")
        st.error(e)


if __name__ == "__main__":
    main()
