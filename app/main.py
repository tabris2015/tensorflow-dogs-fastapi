from app.predictor import Predictor
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class PredictionSchema(BaseModel):
    filename: str
    content_type: str
    prediction: str
    top: dict




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

predictor = Predictor()


@app.get('/')
def get_model_summary():
    stringlist = []
    predictor.model.summary(print_fn=lambda x: stringlist.append(x))
    return {'summary': stringlist}


@app.post('/predict', response_model=PredictionSchema)
def predict_image(file: UploadFile = File(...)):
    prediction = predictor.predict_file(file.file)
    return PredictionSchema(
        filename=file.filename,
        content_type=file.content_type,
        prediction=prediction['label'],
        top=prediction['top']
    )