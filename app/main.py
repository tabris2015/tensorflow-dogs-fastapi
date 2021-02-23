from app.predictor import Predictor
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
from .predictor import Predictor

app = FastAPI()
predictor = Predictor()


@app.get('/')
async def root():
    return {'message': 'Hello World'}

@app.post('/uploadimage')
def create_upload_image(file: UploadFile = File(...)):
    return {
        'filename': file.filename,
        'content-type': file.content_type,
        'prediction': predictor.predict_file(file.file)
        }