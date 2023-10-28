# CS553_ML_Data_Stream_Pipeline
ML data streaming pipeline using Apache Flink
This project is a simple implementation of an ML model inference system. It uses the Apache Flink framework for processing data streams.

## Install Dependencies
Assuming you have <a href="https://docs.anaconda.com/anaconda/install/">Anaconda</a>.
Now, Follow the below steps:

- Step 1: conda create --name cs553
- Step 2: conda activate cs553
- Step 3: pip install -r requirements.txt 

## Install rabitMQ
Using docker: `docker pull rabbitmq:3-management`
Run rabitMQ container and expose ports: `docker run --rm -it -p 15672:15672 -p 5672:5672 rabbitmq:3-management`
RabitMQ management webapp can be accessible at `http://localhost:15672`

## Setup producer and consumer 
Run consumer.py and producer.py in two different terminals. 
Access producer API at http://localhost:5000/produce
APIs available:

```
curl --request POST \
  --url http://localhost:5000/produce \
  --header 'Content-Type: application/json' \
  --data '{"message": "this is an incoming message!"}'
  ```

```
curl --request GET \
  --url http://localhost:5000/produce \
  --header 'Content-Type: application/json'
  ```