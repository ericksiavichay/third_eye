from utils import model_inference
from fastapi import FastAPI, HTTPException, Body, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO

app = FastAPI()


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(BytesIO(content))
    print(content)
    text_result = model_inference(image)
    return JSONResponse(content={"text": text_result}, status_code=200)
