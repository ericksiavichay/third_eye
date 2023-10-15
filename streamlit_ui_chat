import streamlit as st
import requests
from PIL import Image
import io
from elevenlabs import set_api_key
import base64
import asyncio
from deepgram import Deepgram
import sys
from pathlib import Path
import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_openai_response(prompt):
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=50,
            temperature=0.7
        )
        message = response['choices'][0]['text'].strip()
        return message
    except Exception as e:
        return str(e)

async def transcribe_audio(audio_file, DEEPGRAM_API_KEY='5a1672a5d83a9b723a45375eb0077e5c464a8909'):
    # Initialize the Deepgram SDK
    deepgram = Deepgram(DEEPGRAM_API_KEY)
    
    # Set the source
    source = {
      'buffer': audio_file,
      'mimetype': 'audio/mp3'  # or 'audio/wav' based on your input type
    }
  
    # Send the audio to Deepgram and get the response
    response = await asyncio.create_task(
        deepgram.transcription.prerecorded(
            source,
            {
                'smart_format': True,
                'model': 'nova',
            }
        )
    )
  
    # Returning the transcription
    return response["results"]["channels"][0]["alternatives"][0]["transcript"]

def main():
    st.title("thirdeye.")
    set_api_key("1da2bb9407e638f4445d437c8e8770e1")

    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        st.image(img_file_buffer, caption="Uploaded Image.", use_column_width=True)
        byte_img = img_file_buffer.getvalue()
        image_files = {'file': ('image.jpg', byte_img, 'image/jpeg')}
        error_message = "Sorry, didn't work"
        description = error_message

        try:
            image_response = requests.post('http://9ob3xkj0pb.loclx.io/uploadfile', files=image_files)
            image_response.raise_for_status()
            response_json = image_response.json()
            description = response_json.get('text', "No description provided by the API")
        except requests.RequestException as e:
            st.warning("Failed to communicate with the description API.")
            st.error(e)

        st.subheader('Description from API:')
        st.write(description)

        tts_url = "https://api.elevenlabs.io/v1/text-to-speech/D38z5RcWu1voky8WS1ja"
        tts_headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": "1da2bb9407e638f4445d437c8e8770e1"
        }
        tts_data = {
            "text": description,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        try:
            tts_response = requests.post(tts_url, json=tts_data, headers=tts_headers)
            tts_response.raise_for_status()
            
            audio_file = io.BytesIO()
            for chunk in tts_response.iter_content(chunk_size=1024):
                if chunk:
                    audio_file.write(chunk)
            audio_file.seek(0)

            audio_bytes = audio_file.read()
            b64 = base64.b64encode(audio_bytes).decode()
            audio_tag = f'<audio controls autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
            st.markdown(audio_tag, unsafe_allow_html=True)
            
            # Moved "Your turn to respond" here
            st.subheader("Your turn to respond:")
            file_path = Path(__file__).parent / "input.mp3"
            with open(file_path, "rb") as f:
                audio_bytes = io.BytesIO(f.read())

            st.audio(audio_bytes, format='audio/mp3')

            # Handle the audio...
            try:
                # Use asyncio.run() to call your async function and await its result
                transcription = asyncio.run(transcribe_audio(audio_bytes))
                st.subheader("Transcription of your response:")
                st.write(transcription)
            except Exception as e:
                exception_type, exception_object, exception_traceback = sys.exc_info()
                line_number = exception_traceback.tb_lineno
                st.error(f'line {line_number}: {exception_type} - {e}')
            # Get OpenAI GPT-3 response
            openai_response = get_openai_response(transcription)

            # Display GPT-3 response
            st.subheader("Response from OpenAI GPT:")
            st.write(openai_response)

        except requests.RequestException as e:
            st.warning("Failed to communicate with the ElevenLabs API.")
            st.error(e)

if __name__ == "__main__":
    main()
