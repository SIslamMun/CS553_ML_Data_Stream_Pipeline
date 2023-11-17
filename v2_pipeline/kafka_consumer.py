
from kafka import KafkaConsumer
consumer = KafkaConsumer('testoutput', group_id='output')
for msg in consumer:
    print (msg.value)