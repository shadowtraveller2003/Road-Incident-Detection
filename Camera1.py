import cv2
from detection import AccidentDetectionModel
import numpy as np

# Initialize the model
model = AccidentDetectionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

def check_single_image(frame):
    if frame is None:
        print("Error: Could not load image.")
        return
    
    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize the image to the input size expected by the model
    roi = cv2.resize(rgb_frame, (250, 250))

    # Predict accident status
    pred, prob = model.predict_accident(roi[np.newaxis, :, :, :])
    
    if pred == "Accident":
        prob = round(prob[0][0] * 100, 2)

        # Display the prediction on the image
        cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"{pred} {prob}%", (20, 30), font, 1, (255, 255, 0), 2)

    # Show the image with prediction
    #cv2.imshow('Image', frame)
    
    # Wait indefinitely until 'q' is pressed to close all windows
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
