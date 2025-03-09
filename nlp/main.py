from typing import Union
import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import processors

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import numpy as np
import io
from tempfile import NamedTemporaryFile
import uuid

import insightface
from insightface.app import FaceAnalysis

app = FastAPI()

# Static files 설정
app.mount("/static", StaticFiles(directory="static"), name="static")

# STEP 1: Import modules for face recognition
import insightface
from insightface.app import FaceAnalysis

# Initialize FaceAnalysis globally
face_app = FaceAnalysis()
face_app.prepare(ctx_id=0, det_size=(640, 640))

# MediaPipe 모델 설정
base_options_cls = python.BaseOptions(model_asset_path='models/efficientnet_lite0.tflite')
options_cls = vision.ImageClassifierOptions(base_options=base_options_cls, max_results=3)
classifier = vision.ImageClassifier.create_from_options(options_cls)

# STEP 2: Create an ObjectDetector object.
base_options_det = python.BaseOptions(model_asset_path='models\\efficientdet_lite2.tflite')
options_det = vision.ObjectDetectorOptions(base_options=base_options_det, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options_det)

# Ensure the static directory exists
os.makedirs('static', exist_ok=True)

@app.get("/")
async def redirect_to_home():
    return RedirectResponse(url="/home")

@app.get("/items/{item_id}")
def read_item(item_id: int = 1, q: Union[str, None] = "test"):
    return {"item_id": item_id, "q": q}

@app.post("/img_cls")
async def img_cls(image: UploadFile = File(...)):
    contents = await image.read()
    filename = "static/input_image.jpg"
    with open(filename, "wb") as f:
        f.write(contents)

    mp_image = mp.Image.create_from_file(filename)
    classification_result = classifier.classify(mp_image)

    top_category = classification_result.classifications[0].categories[0]
    
    # 결과 이미지를 저장
    result_image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    result_filename = "static/result_image.jpg"
    cv2.imwrite(result_filename, result_image)

    return RedirectResponse(url=f"/img_cls_result?message=이미지 분류 요청이 성공적으로 처리되었습니다&top_category={top_category.category_name}&score={top_category.score}", status_code=303)

@app.get("/img_cls_result")
async def img_cls_result(message: str, top_category: str, score: float):
    return HTMLResponse(content=open('static/img_cls_result.html', encoding='utf-8').read().format(message=message, top_category=top_category, score=score), status_code=200)

@app.post("/detect")
async def detect(
    image: UploadFile = File(...)
):
    contents = await image.read()
    filename = "static/input_image.jpg"
    with open(filename, "wb") as f:
        f.write(contents)

    image_np = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
    detection_result = detector.detect(mp_image)

    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = (int(bbox.origin_x), int(bbox.origin_y))
        end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
        cv2.rectangle(image_np, start_point, end_point, (0, 255, 0), 2)
        label = f"{detection.categories[0].category_name}: {detection.categories[0].score:.2f}"
        cv2.putText(image_np, label, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    result_filename = "static/result_image.jpg"
    cv2.imwrite(result_filename, image_np)

    detections_str = ", ".join([f"{d.categories[0].category_name} ({d.categories[0].score:.2f})" for d in detection_result.detections])

    return RedirectResponse(url=f"/detect_result?message=객체 탐지 요청이 성공적으로 처리되었습니다&detections={detections_str}", status_code=303)

@app.get("/detect_result")
async def detect_result(message: str, detections: str):
    return HTMLResponse(content=open('static/detect_result.html', encoding='utf-8').read().format(message=message, detections=detections), status_code=200)

@app.get("/home")
async def home():
    return HTMLResponse(content=open('static/home.html', encoding='utf-8').read(), status_code=200)

@app.get("/img_cls")
async def img_cls_form():
    return HTMLResponse(content="""
    <html>
        <head>
            <title>Image Classification</title>
            <link rel="stylesheet" type="text/css" href="/static/style.css">
        </head>
        <body>
            <div class="container">
                <h1>Upload Image for Classification</h1>
                <form action="/img_cls" method="post" enctype="multipart/form-data">
                    <input type="file" name="image" accept="image/*">
                    <input type="submit" value="Classify Image">
                </form>
            </div>
        </body>
    </html>
    """, status_code=200)

@app.get("/detect")
async def detect_form():
    return HTMLResponse(content="""
    <html>
        <head>
            <title>Object Detection</title>
            <link rel="stylesheet" type="text/css" href="/static/style.css">
        </head>
        <body>
            <div class="container">
                <h1>Upload Image for Object Detection</h1>
                <form action="/detect" method="post" enctype="multipart/form-data">
                    <input type="file" name="image" accept="image/*">
                    <input type="submit" value="Detect Objects">
                </form>
            </div>
        </body>
    </html>
    """, status_code=200)

@app.get("/result")
async def result_page(image_filename: str, message: str, detections: str, image_url: str):
    return HTMLResponse(content=f"""
    <html>
        <head>
            <title>Result</title>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .container {{ max-width: 800px; margin: auto; text-align: center; }}
                .result-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .result-table th, .result-table td {{ border: 1px solid #ddd; padding: 8px; }}
                .result-table th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; }}
                button {{ margin-top: 20px; padding: 10px 20px; font-size: 16px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Result</h1>
                <table class="result-table">
                    <tr><th>Message</th><td>{message}</td></tr>
                    <tr><th>Image Filename</th><td>{image_filename}</td></tr>
                    <tr><th>Detections</th><td>{detections}</td></tr>
                </table>
                <img src="{image_url}" alt="Result Image">
                <button onclick=\"window.location.href='/home'\">Return to Home</button>
            </div>
        </body>
    </html>
    """, status_code=200)

@app.get("/face_recognition")
async def face_recognition_form():
    return HTMLResponse(content="""
    <html>
        <head>
            <title>Face Recognition</title>
            <link rel="stylesheet" type="text/css" href="/static/style.css">
        </head>
        <body>
            <div class="container">
                <h1>Upload Images for Face Recognition</h1>
                <form action="/face_recognition" method="post" enctype="multipart/form-data">
                    <input type="file" name="image1" accept="image/*">
                    <input type="file" name="image2" accept="image/*">
                    <input type="submit" value="Compare Faces">
                </form>
            </div>
        </body>
    </html>
    """, status_code=200)

@app.post("/face_recognition")
async def face_recognition(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    contents1 = await image1.read()
    contents2 = await image2.read()
    
    # 파일을 고정된 이름으로 저장
    filename1 = "static/input_image1.jpg"
    filename2 = "static/input_image2.jpg"
    with open(filename1, "wb") as f1, open(filename2, "wb") as f2:
        f1.write(contents1)
        f2.write(contents2)

    image_np1 = cv2.imdecode(np.frombuffer(contents1, np.uint8), cv2.IMREAD_COLOR)
    image_np2 = cv2.imdecode(np.frombuffer(contents2, np.uint8), cv2.IMREAD_COLOR)

    faces1 = face_app.get(image_np1)
    faces2 = face_app.get(image_np2)

    for face in faces1:
        bbox = face.bbox.astype(int)
        cv2.rectangle(image_np1, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    for face in faces2:
        bbox = face.bbox.astype(int)
        cv2.rectangle(image_np2, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    if len(faces1) == 1 and len(faces2) == 1:
        feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
        feat2 = np.array(faces2[0].normed_embedding, dtype=np.float32)
        similarity = np.dot(feat1, feat2.T)
    else:
        similarity = None

    # 결과 이미지를 고정된 이름으로 저장
    result_filename1 = "static/result_image1.jpg"
    result_filename2 = "static/result_image2.jpg"
    cv2.imwrite(result_filename1, image_np1)
    cv2.imwrite(result_filename2, image_np2)

    similarity_str = f"{similarity:.2f}" if similarity is not None else "N/A"

    return RedirectResponse(url=f"/face_recognition_result?similarity={similarity_str}", status_code=303)

@app.get("/face_recognition_result")
async def face_recognition_result(similarity: str):
    return HTMLResponse(content=open('static/face_recognition_result.html', encoding='utf-8').read().format(similarity=similarity), status_code=200)
