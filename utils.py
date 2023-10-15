from fastapi import FastAPI, HTTPException, Body, File, UploadFile
import numpy as np
import requests
import time
from PIL import Image
from transformers import pipeline
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from fastapi.responses import JSONResponse
from io import BytesIO
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# from ultralytics import YOLO
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

app = FastAPI()
# model = Pix2StructForConditionalGeneration.from_pretrained(
#     "google/pix2struct-textcaps-base"
# ).to("cuda")
# processor = Pix2StructProcessor.from_pretrained("google/pix2struct-textcaps-base")

pipe = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

# image_to_text = YOLO("yolov8n.pt")


# def yolo_single_frame(frame):
#     some_return = model.predict(frame, show=True, stream=True)
#     return some_return


# def stream_object_detection():
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         exit()

#     while True:
#         # Read a frame from the webcam
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Couldn't read frame.")
#             break

#         # Use YOLO to predict on the frame directly
#         some_return = model.predict(frame, show=True, stream=True)
#         #

#         # Display the frame
#         cv2.imshow("YOLO Real-time Detection", frame)

#         # Wait for keypress
#         key = cv2.waitKey(2)  # you'll have to play around with the amount of time

#         # Check if the pressed key is the spacebar
#         if key == 32:
#             # Capture user prompt
#             user_prompt = input("Enter your prompt: ")

#             inputs = processor(frame, return_tensors="pt", padding="max_length")
#             out = image_to_text.generate(**inputs)
#             generated_text = processor.decode(out[0], skip_special_tokens=True)

#             print("-" * 50)
#             print(f"Generated Text: {generated_text}")
#             print("-" * 50)
#             cap.release()
#             cv2.destroyAllWindows()
#             return generated_text, frame


# def get_mediapipe_outputs(image, detection_result):
#     """Draws bounding boxes on the input image and return it.
#     Args:
#         image: The input RGB image.
#         detection_result: The list of all "Detection" entities to be visualize.
#     Returns:
#         Image with bounding boxes.
#     """
#     for detection in detection_result.detections:
#         # Draw bounding_box
#         bbox = detection.bounding_box
#         start_point = bbox.origin_x, bbox.origin_y
#         end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
#         cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

#         # Draw label and score
#         category = detection.categories[0]
#         category_name = category.category_name
#         probability = round(category.score, 2)
#         result_text = category_name + " (" + str(probability) + ")"
#         text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
#         cv2.putText(
#             image,
#             result_text,
#             text_location,
#             cv2.FONT_HERSHEY_PLAIN,
#             FONT_SIZE,
#             TEXT_COLOR,
#             FONT_THICKNESS,
#         )

#     return image


def get_yolo_outputs(image):
    """
    Takes in an image and returns a dictionary str: int
    """
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    classes = open("coco.names").read().strip().split("\n")
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # cap = cv2.VideoCapture(0)

    # _, frame = cap.read()
    frame = np.array(image)
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
    )
    net.setInput(blob)
    outs = net.forward(output_layers)
    print("OUTS", outs)

    # Information on screen (class id, confidence, bounding box coordinates)
    class_ids = []
    confidences = []
    boxes = []

    counts = {}
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            label = str(classes[class_id])

            if confidence > 0.9:
                # center_x = int(detection[0] * width)
                # center_y = int(detection[1] * height)
                # w = int(detection[2] * width)
                # h = int(detection[3] * height)
                # x = int(center_x - w / 2)
                # y = int(center_y - h / 2)
                # boxes.append([x, y, w, h])
                # confidences.append(float(confidence))
                # class_ids.append(class_id)
                if label in counts:
                    counts[label] += 1
                else:
                    counts[label] = 1

    print(counts)
    return counts

    # indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # for i in range(len(boxes)):
    #     if i in indexes:
    #         label = str(classes[class_ids[i]])
    #         color = colors[i]
    #         cv2.rectangle(
    #             frame,
    #             (boxes[i][0], boxes[i][1]),
    #             (boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]),
    #             color,
    #             2,
    #         )
    #         cv2.putText(
    #             frame,
    #             label,
    #             (boxes[i][0], boxes[i][1] - 10),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5,
    #             color,
    #             2,
    #         )

    # cap.release()
    # cv2.destroyAllWindows()


def model_inference(input_image, input_prompt=""):
    image = input_image

    # inputs = processor(images=image, return_tensors="pt").to("cuda")
    start_time = time.time()
    # predictions = model.generate(**inputs)
    # yolo_returns = yolo_single_frame(image)
    end_time = time.time()
    response = pipe(image)

    # response = processor.decode(predictions[0], skip_special_tokens=True)
    print(response)
    print(end_time - start_time)
    return response[0]["generated_text"]
