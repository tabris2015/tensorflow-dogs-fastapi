from app.predictor import Predictor
from fastapi import FastAPI, File, UploadFile

app = FastAPI()
predictor = Predictor()


@app.post('/predict')
def predict_image(file: UploadFile = File(...)):
    prediction = predictor.predict_file(file.file)
    return {
        'filename': file.filename,
        'content-type': file.content_type,
        'prediction': prediction['label'],
        'top': prediction['top']
        }