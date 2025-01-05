from keras.models import model_from_json, Sequential
import numpy as np
from keras.utils import get_custom_objects

# Register the Sequential class
get_custom_objects().update({'Sequential': Sequential})

class AccidentDetectionModel:
    class_names = ['Accident', 'No Accident']

    def __init__(self, model_json_file, model_weights_file):
        # Load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # Load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()

    def predict_accident(self, img):
        preds = self.loaded_model.predict(img)
        return AccidentDetectionModel.class_names[np.argmax(preds)], preds
