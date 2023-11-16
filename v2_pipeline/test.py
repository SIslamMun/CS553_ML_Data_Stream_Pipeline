from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors import RMQSource,RMQConnectionConfig
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream.connectors.rabbitmq import RMQSource, RMQConnectionConfig

env = StreamExecutionEnvironment.get_execution_environment()
env.add_jars("file:///home/mun/Illinois_Tech_MS/fall23_iit/CS553/project/code/flink-connector-rabbitmq-1.16.2.jar")

# print(env)
rabbitmq_config = RMQConnectionConfig.Builder() \
    .setHost("localhost") \
    .build()

# rabbitmq_source = RMQSource(rabbitmq_config, "hello", True, None)

# env.add_source(source).print()

# env.execute("RabbitMQ Flink Example")

# connection_config = RMQConnectionConfig.Builder() \
#     .set_host("localhost") \
#     .build()

# stream = env \
#     .add_source(RMQSource(
#         connection_config,
#         "hello",
#         True,
#         SimpleStringSchema(),
#     )) \
#     .set_parallelism(1)

# connection_config = {
#     "host": "localhost",
# }
# # rmq_connection_config_obj = RMQSource.convert_rmq_config(connection_config)

# rmq_connection_config = RMQConnectionConfig.Builder() \
#     .setHost("localhost") \
#     .setPort(5672) \
#     .build()

# # Create a RabbitMQ source
# source = RMQSource(connection_config=rmq_connection_config,
#                            queue_name='hello',
#                            use_correlation_id= True,
#                            deserialization_schema=SimpleStringSchema())
