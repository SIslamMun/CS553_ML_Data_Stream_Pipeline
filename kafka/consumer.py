from kafka import KafkaConsumer
import cv2
import numpy as np
# This is a sample one, to verify that a consumer can pick images from topic. This will be replaced with flink consumer.
def consume_images(consumer, topic_name):
    for message in consumer:
        img_bytes = message.value
        # Convert bytes to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print("Recieved image!")

# Kafka Consumer Configuration
bootstrap_servers = 'localhost:9092'
topic = 'image_topic'

# Create Kafka Consumer
consumer = KafkaConsumer(topic, bootstrap_servers=bootstrap_servers)

# Consume images from Kafka topic
consume_images(consumer, topic)

