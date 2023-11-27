from kafka import KafkaProducer
import cv2
import numpy as np
import os
import time

def produce_image(producer, topic_name, image_path):
    image = cv2.imread(image_path)
    # Convert image to bytes
    _, img_encoded = cv2.imencode('.jpg', image)
    img_bytes = img_encoded.tobytes()
    producer.send(topic_name, img_bytes)
    producer.flush()

# Kafka Producer Configuration
bootstrap_servers = 'localhost:9092'
topic = 'image_topic'

# Create Kafka Producer
producer = KafkaProducer(bootstrap_servers=bootstrap_servers)

# Directory to monitor for new images
directory_to_watch = '/kafka/images'

while True:
    # Get list of files in the directory
    files = os.listdir(directory_to_watch)

    # Filter image files
    image_files = [file for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if image_files:
        for image_file in image_files:
            image_path = os.path.join(directory_to_watch, image_file)
            
            # Produce image to Kafka topic
            produce_image(producer, topic, image_path)
            
            # Delete the image file after pushing to Kafka
            os.remove(image_path)
            print(f"Found new image at {image_path} - pushed to kafka")

    # Sleep for a while before checking again
    time.sleep(5)  
