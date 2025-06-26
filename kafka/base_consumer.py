from confluent_kafka import Consumer, KafkaException
import sys

class KafkaMessageConsumer:
    def __init__(self, config, topic, message_handler):
        self.consumer = Consumer(config)
        self.topic = topic
        self.message_handler = message_handler

    def start(self):
        self.consumer.subscribe([self.topic])
        print(f"Subscribed to topic: {self.topic}")
        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    print(f"Consumer error: {msg.error()}")
                    continue
                self.message_handler(msg.value())
        except Exception as e:
            print(f"Exception occurred: {e}")
        finally:
            self.consumer.close()
            print("Consumer closed.")

