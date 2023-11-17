from kafka import KafkaProducer
import json
import cv2
import base64

producer = KafkaProducer(
  bootstrap_servers='localhost:9092',
  # value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
# ...


path = "data/single_person.mp4"

cap = cv2.VideoCapture(path)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    image = cv2.imencode('.jpg', frame)[1]
    encoded_img = base64.b64encode(image)

    # producer.send('test', {"data": str(encoded_img)})
    producer.send('test',encoded_img)

    
# image = cv2.imread("data/images/test.png")
# data = cv2.imencode('.jpeg', image)[1].tobytes()
# encoded_img = base64.b64encode(data)

# producer.send('test', {"data": str(encoded_img)})
# producer.send('test',encoded_img)
# producer.send('test',encoded_img)
# producer.send('test',encoded_img)
# producer.send('test',encoded_img)
producer.close()