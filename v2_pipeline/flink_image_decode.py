import argparse
import logging
import sys
import cv2
import numpy as np
import base64
# from pyflink.common import WatermarkStrategy, Encoder, Types
from pyflink.datastream import StreamExecutionEnvironment, RuntimeExecutionMode
# from pyflink.datastream.connectors.file_system import FileSource, StreamFormat, FileSink, OutputFileConfig, RollingPolicy
from pyflink.table import StreamTableEnvironment
from pyflink.datastream.functions import RuntimeContext, MapFunction
from pyflink.datastream.connectors.rabbitmq import RMQSource, RMQConnectionConfig


from pyflink.common import SerializationSchema, DeserializationSchema
from pyflink.datastream.functions import SinkFunction, SourceFunction
from pyflink.java_gateway import get_gateway




def flink_streaming_engine():


    # input_path = "test.png"
    env = StreamExecutionEnvironment.get_execution_environment()
    # t_env = StreamTableEnvironment.create(stream_execution_environment=env)
    # # Add the RabbitMQ connector library to the pipeline.jars configuration
    # jar_url = "file:///home/mun/Illinois_Tech_MS/fall23_iit/CS553/project/code/flink-connector-rabbitmq_2.11-1.13.1.jar"
    # t_env.get_config().get_configuration().set_string("pipeline.jars", jar_url)
    # Setting up RabbitMQ configuration
    rabbitmq_config = RMQConnectionConfig.Builder() \
        .setHost("localhost") \
        .setVirtualHost("/") \
        .setPort(5672) \
        .build()

    rabbitmq_source = RMQSource(rabbitmq_config, "hello", True, None)
    env.set_runtime_mode(RuntimeExecutionMode.BATCH)
    # # write all the data to one file
    

    env.set_parallelism(1)
    # t_env.set_parallelism(1)


    

    # # data_stream = env.add_source(rabbitmq_source)
    # ds = env.from_collection(input_path)

    # ds.print()

    
    # person detection
    # ds = ds.flat_map(run,output_type=Types.STRING())
    # ds = data_stream.map(run,output_type=Types.STRING())

    # # define the sink

    
    # ds.print()
    # data_stream.print()
    # ds.sink_to(
    #     sink=FileSink.for_row_format(
    #         base_path='data/output',
    #         encoder=Encoder.simple_string_encoder())
    #     .with_output_file_config(
    #         OutputFileConfig.builder()
    #         .build())
    #     .with_rolling_policy(RollingPolicy.default_rolling_policy())
    #     .build()
    # )

    # submit for execution
    env.execute()




if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        dest='input',
        required=False,
        help='Input file to process.')
    parser.add_argument(
        '--output',
        dest='output',
        required=False,
        help='Output file to write results to.')

    argv = sys.argv[1:]
    known_args, _ = parser.parse_known_args(argv)

    flink_streaming_engine()

