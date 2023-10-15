from utils import model_inference, get_yolo_outputs
from fastapi import FastAPI, HTTPException, Body, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import openai
import os
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
OPENAI_API_URL = 'https://api.openai.com/v1/chat/completions'
app = FastAPI()

def call_openai_api(prompt: str): 
    messages = [ {"role": "user", "content": prompt} ]
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json',
    }
    chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
      
    reply = chat.choices[0].message.content
    
    # prompt = reply + ' ' + 'Just provide highlight on a higher level as you are instructing a user who is doing yoga'
    # messages = [ {"role": "system", "content": prompt} ]
    # chat = openai.ChatCompletion.create(
    #         model="gpt-3.5-turbo", messages=messages
    #     )
    # reply = chat.choices[0].message.content
    print(reply)
    return reply

def generate_text(prompt: dict):
    try:
        # print(prompt)
        # print(prompt['pose'])
        response_text = True
        deviation = prompt['objects']
        prompt_text = "Assume you are AI assistant to a blind person. User is trying to understand what is happening around him.\
              You are given a list of object presents around the blind person in form of objects mapped to number of objects along with a action recognition on the frame \
            Just summarize what is around him in 2 small lines"
        final_prompt = prompt_text + ' ' + str(deviation) + "Action : "+prompt["text"]
        print(final_prompt)
        print ('')
        print ('')
        response_text = call_openai_api(final_prompt)
        return response_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(BytesIO(content))
    print(content)
    text_result = model_inference(image)
    obj_det = get_yolo_outputs(image)
    reply = generate_text({"text":text_result,"objects":obj_det})
    return JSONResponse(content={"text": reply}, status_code=200)
