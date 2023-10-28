from flask import Flask, request, jsonify

import pika

app = Flask(__name__)

# RabbitMQ connection parameters
rabbitmq_host = 'localhost'

@app.route('/produce', methods=['POST'])
def produce_message():
    try:
        message = request.json['message']
        
        # Connect to RabbitMQ server
        connection = pika.BlockingConnection(pika.ConnectionParameters(rabbitmq_host))
        channel = connection.channel()

        # Create a queue named 'queue1'
        channel.queue_declare(queue='queue1')

        # Publish the message to the 'queue1' queue
        channel.basic_publish(exchange='', routing_key='queue1', body=message)

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
