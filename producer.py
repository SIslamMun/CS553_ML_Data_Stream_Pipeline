from flask import Flask, request, jsonify
import configparser

import pika

app = Flask(__name__)


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
        print(f"Queue '{queue}' already exists.")
    except pika.exceptions.ChannelClosed as e:
        # Queue does not exist, create it
        channel.queue_declare(queue=queue)
        print(f"Queue '{queue}' created.")

@app.route('/produce', methods=['POST'])
def produce_message():
    try:
        message = request.json['message']
        

        channel.basic_publish(exchange='', routing_key=queue, body=message)

        # Close the connection
        connection.close()

        return jsonify({"status": "Message sent successfully"})
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