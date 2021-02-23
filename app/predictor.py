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

    def predict_image(self, image, n_top=3):
        pred = self.model.predict(image.reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 3))
        top_labels = {}
        if len(self.labels) >= n_top:
            top_labels_ids = np.flip(np.argsort(pred, axis=1)[0, -3:])
            for label_id in top_labels_ids:
                top_labels[self.labels[label_id]] = pred[0,label_id].item()
        pred_label = self.labels[np.argmax(pred)]
        print(top_labels)
        return {'label': pred_label, 'top': top_labels}

    def predict_file(self, file, n_top=3):
        img = np.array(Image.open(file).resize((self.IMG_SIZE,self.IMG_SIZE)), dtype=np.float32)
        return self.predict_image(img, n_top)