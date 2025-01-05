import pika
import cv2
import base64
import numpy as np
from Camera1 import check_single_image  # Import the function to process the image
from sample import model  # Import the model from sample.py

# Set up the connection and channel for RabbitMQ
connection = pika.BlockingConnection(pika.ConnectionParameters('10.1.3.2'))
channel = connection.channel()

channel.exchange_declare(exchange='frame', exchange_type='direct')
channel.queue_declare(queue='frames_queue')
channel.queue_bind(exchange='frame', queue='frames_queue')

def callback(ch, method, properties, body):
    # Decode the image received from RabbitMQ
    jpg_original = base64.b64decode(body)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)

    # 1. Pass the image to the first model (Accident Detection Model in Camera1)
    #check_single_image(img)  # This will process the image and show the accident prediction

    # 2. Pass the image to the second model (from sample.py)
    # Convert the image to RGB for prediction (because OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    label = model.predict(image=image_rgb)['label']
    
    # Add the label from the second model (Violence Prediction) to the image
    cv2.putText(img, f"Violence: {label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # 3. Display both the accident label (from Camera1) and violence label (from sample.py)
    # The check_single_image function already displays the accident label on the image.
    # The violence label is now added to the image.

    # Show the image with both labels
    cv2.imshow('Image with Accident and Violence Predictions', img)

    # Check if 'q' is pressed to close the image window and stop consuming
    if cv2.waitKey(1) & 0xFF == ord('q'):
        ch.basic_ack(delivery_tag=method.delivery_tag)
        cv2.destroyAllWindows()
        connection.close()
        return

    # Acknowledge that the message was processed (if auto_ack is False)
    ch.basic_ack(delivery_tag=method.delivery_tag)

# Start consuming messages from RabbitMQ
print('Waiting for messages...')
channel.basic_consume(queue='frames_queue', on_message_callback=callback, auto_ack=False)
channel.start_consuming()
