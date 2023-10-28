import pika

def callback(ch, method, properties, body):
    print(f"Received: {body}")

# Connect to RabbitMQ server
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Create a queue named 'queue1'
channel.queue_declare(queue='queue1')

# Set up a consumer callback function
channel.basic_consume(queue='queue1', on_message_callback=callback, auto_ack=True)

print("Waiting for messages. To exit, press CTRL+C")
channel.start_consuming()
