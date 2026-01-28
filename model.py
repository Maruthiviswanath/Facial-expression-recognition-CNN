import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json, Sequential

# 🔴 VERY IMPORTANT for TF1 models
tf.compat.v1.disable_eager_execution()

class ERModel(object):

    EMOTIONS_LIST = [
        "Angry", "Disgust", "Fear",
        "Happy", "Neutral", "Sad", "Surprise"
    ]

    def __init__(self, model_json_file, model_weights_file):

        # 🔹 Create graph & session ONCE
        self.graph = tf.compat.v1.get_default_graph()
        self.session = tf.compat.v1.Session(graph=self.graph)

        with self.graph.as_default():
            with self.session.as_default():

                with open(model_json_file, "r") as json_file:
                    loaded_model_json = json_file.read()

                self.loaded_model = model_from_json(
                    loaded_model_json,
                    custom_objects={"Sequential": Sequential}
                )

                self.loaded_model.load_weights(model_weights_file)

                self.loaded_model.compile(
                    optimizer="adam",
                    loss="categorical_crossentropy",
                    metrics=["accuracy"]
                )

        print("✅ Emotion model loaded successfully")

    def predict_emotion(self, img):
        with self.graph.as_default():
            with self.session.as_default():

                preds = self.loaded_model.predict(img, verbose=0)[0]

                idx = int(np.argmax(preds))
                emotion = ERModel.EMOTIONS_LIST[idx]
                confidence = float(preds[idx] * 100)

                return emotion, confidence
