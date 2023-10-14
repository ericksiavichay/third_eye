from fastapi import FastAPI, HTTPException, Body,  File, UploadFile
import httpx
import numpy as np
import requests
import time
from PIL import Image
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from fastapi.responses import JSONResponse
from io import BytesIO


app = FastAPI()
model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-base").to("cuda")
processor = Pix2StructProcessor.from_pretrained("google/pix2struct-textcaps-base")

def model_inference(content):
    image = Image.open(BytesIO(content))
    

    inputs = processor(images=image, return_tensors="pt").to("cuda")
    start_time = time.time()
    predictions = model.generate(**inputs)
    end_time = time.time()

    response = processor.decode(predictions[0], skip_special_tokens=True)
    print(response)
    print(end_time-start_time)
    return response
    
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    content = await file.read()
    print(content)
    text_result = model_inference(content)
    return JSONResponse(content={"text": text_result}, status_code=200)