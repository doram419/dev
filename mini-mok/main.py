from typing import Union
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
import shutil
import os

app = FastAPI()

# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/uploadfile/")
async def create_upload_file(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    file_location1 = "static/result1.jpg"
    file_location2 = "static/result2.jpg"
    with open(file_location1, "wb") as buffer1:
        shutil.copyfileobj(file1.file, buffer1)
    with open(file_location2, "wb") as buffer2:
        shutil.copyfileobj(file2.file, buffer2)
    return RedirectResponse(url="/result", status_code=303)

@app.get("/result", response_class=HTMLResponse)
async def read_result(request: Request):
    return templates.TemplateResponse("result.html", {"request": request, "filename1": "result1.jpg", "filename2": "result2.jpg"})

@app.get("/face/{face_id}", response_class=HTMLResponse)
async def read_face(request: Request, face_id: int):
    return templates.TemplateResponse("face.html", {"request": request, "face_id": face_id})