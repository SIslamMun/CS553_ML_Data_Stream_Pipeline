import base64
from flask import Flask, request, jsonify
import configparser
import logging
import pika
import sys

app = Flask(__name__)

stream_handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
stream_handler.setFormatter(log_formatter)
log_level = logging.INFO #TODO later
logger = logging.getLogger("pipeline")
logger.setLevel(log_level)
logger.addHandler(stream_handler)

def load_config(config_file="config.ini"):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

config = load_config()

rabbitmq_host = config.get("rabitmq", "host")
queue = config.get("rabitmq", "queue")

# Connect to RabbitMQ server
connection = pika.BlockingConnection(pika.ConnectionParameters(rabbitmq_host))
channel = connection.channel()

def createQueue(channel, queue):
    try:
        channel.queue_declare(queue=queue, passive=True)
        logger.info(f"Queue '{queue}' already exists.")
    except pika.exceptions.ChannelClosed as e:
        # Queue does not exist, create it
        channel.queue_declare(queue=queue)
        logger.info(f"Queue '{queue}' created.")

@app.route('/produce', methods=['POST'])
def produce_message():
    try:
        # message = request.json['message']
        message = request.files.get('image')
        if message:
            image_data = base64.b64encode(message.read()).decode()
            channel.basic_publish(exchange='', routing_key=queue, body=image_data)
        # Close the connection
            connection.close()
            logger.info("Pushed image to the queue")
            return jsonify({"status": "Message sent successfully"})
        else:
            logger.info("Image not provided in the request input")
            return jsonify({"status": "request failed"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/produce', methods=['GET'])
def producer_status():
    try:
        return jsonify({"Status":"RabitMQ is running"})
    except:
        return jsonify({"Status":"RabitMQ is down"})


if __name__ == '__main__':
    app.run(debug=True)
    createQueue(channel, queue)