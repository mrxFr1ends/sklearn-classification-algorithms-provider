import pika
import json


class RabbitService:
    def __init__(self, host, port, user, password):
        credentials = pika.PlainCredentials(user, password)
        connection_parameters = pika.ConnectionParameters(
            host=host, port=port, credentials=credentials)
        self.connection = pika.BlockingConnection(connection_parameters)
        self.channel = self.connection.channel()

    def __del__(self):
        self.connection.close()

    def add_topic(self, topic_name):
        self.channel.queue_declare(queue=topic_name)
        self.channel.queue_bind(topic_name, topic_name, topic_name)

    def add_exchange(self, exchange_name, exchange_type):
        self.channel.exchange_declare(
            exchange=exchange_name,
            exchange_type=exchange_type)

    def send_message(self, topic_name, message):
        self.channel.basic_publish(
            exchange=topic_name,
            routing_key=topic_name,
            body=json.dumps(message)
        )

    def __on_message_received(self, channel, method, properties, body):
        if self.callback_func is not None:
            self.callback_func(channel, method, properties, body)
        self.last_message = body
        self.channel.stop_consuming()

    def consume_message(self, topic_name, callback_func=None, auto_ack=False):
        self.callback_func = callback_func
        self.start_consuming(topic_name, self.__on_message_received, auto_ack)
        return self.last_message

    def start_consuming(self, topic_name, callback_func, auto_ack=False):
        self.channel.basic_consume(
            queue=topic_name, auto_ack=auto_ack, on_message_callback=callback_func)
        self.channel.start_consuming()
