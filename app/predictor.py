import numpy as np
from PIL import Image
import tensorflow as tf

class Predictor:
    BREED_FILE = 'app/breeds.txt'
    MODEL_FILE = 'app/models/best_local1.h5'
    IMG_SIZE = 224
    def __init__(self):
        self.labels = []
        with open(self.BREED_FILE, 'r') as f:
            for line in f:
                self.labels.append(line.strip())
        
        self.model = tf.keras.models.load_model(self.MODEL_FILE)

    def predict_image(self, image):
        pred = self.model.predict(image.reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 3))
        
        if len(self.labels) > 3:
            top_three = np.argsort(pred, axis=1)[0, -3:]
            for label in top_three:
                print(f'{self.labels[label]}: {pred[0,label]}')
        pred_label = self.labels[np.argmax(pred)]
        return pred_label

    def predict_file(self, file):
        img = np.array(Image.open(file).resize((self.IMG_SIZE,self.IMG_SIZE)), dtype=np.float32)
        return self.predict_image(img)