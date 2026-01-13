import json
from kafka import KafkaProducer
from config.settings import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC_RAW


def get_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )


def send_noise_data(producer, data: list):
    for record in data:
        producer.send(KAFKA_TOPIC_RAW, record)
    producer.flush()
