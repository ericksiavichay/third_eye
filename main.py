from fastapi import FastAPI, HTTPException, Body, File, UploadFile
import numpy as np
import requests
import time
from PIL import Image
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from fastapi.responses import JSONResponse
from io import BytesIO
import cv2
from ultralytics import YOLO


app = FastAPI()
model = Pix2StructForConditionalGeneration.from_pretrained(
    "google/pix2struct-textcaps-base"
).to("cuda")
processor = Pix2StructProcessor.from_pretrained("google/pix2struct-textcaps-base")
image_to_text = YOLO("yolov8n.pt")


def stream_object_detection():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame.")
            break

        # Use YOLO to predict on the frame directly
        some_return = model.predict(frame, show=True, stream=True)
        #

        # Display the frame
        cv2.imshow("YOLO Real-time Detection", frame)

        # Wait for keypress
        key = cv2.waitKey(2)  # you'll have to play around with the amount of time

        # Check if the pressed key is the spacebar
        if key == 32:
            # Capture user prompt
            user_prompt = input("Enter your prompt: ")

            # Save the current frame for pix2struct processing
            # cv2.imwrite("temp_frame.jpg", frame)

            # # Process the image with pix2struct
            # with open("temp_frame.jpg", "rb") as image_file:
            #     image = image_file.read()

            inputs = processor(frame, return_tensors="pt", padding="max_length")
            out = image_to_text.generate(**inputs)
            generated_text = processor.decode(out[0], skip_special_tokens=True)

            print("-" * 50)
            print(f"Generated Text: {generated_text}")
            print("-" * 50)
            cap.release()
            cv2.destroyAllWindows()
            return generated_text, frame
            break

        # Exit loop when 'q' is pressed
        elif key == ord("q"):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()


def model_inference(input_image, input_prompt=""):
    image = input_image

    inputs = processor(images=image, return_tensors="pt").to("cuda")
    start_time = time.time()
    predictions = model.generate(**inputs)
    end_time = time.time()

    response = processor.decode(predictions[0], skip_special_tokens=True)
    print(response)
    print(end_time - start_time)
    return response


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(BytesIO(content))
    print(content)
    text_result = model_inference(image)
    return JSONResponse(content={"text": text_result}, status_code=200)
