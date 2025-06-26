from kafka.base_consumer import KafkaMessageConsumer
from config.config_loader import Config

def handle_video_message(msg):
    print("Received message:", msg.decode('utf-8'))

config = Config()
kafka_config = {
    'bootstrap.servers': config['kafka']['bootstrap_servers'],
    'group.id': config['kafka']['group_id'],
    'auto.offset.reset': config['kafka']['auto_offset_reset']
}
topic = config['topic']['frames_consumer']

def start_video_consumer():
    consumer = KafkaMessageConsumer(kafka_config, topic, handle_video_message)
    consumer.start()

if __name__ == "__main__":
    start_video_consumer()