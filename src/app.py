import warnings
warnings.filterwarnings('ignore')
import time
import json
import traceback
from rabbit_service import RabbitService
from parsers import MainParser
from config import RabbitConfig, ProviderConfig, MainTopic, REGISTER_MESSAGE, REQUEST_HANDLER_MAP, ERROR_MESSAGE


def send_register_message(topic_name):
    service.send_message(topic_name, REGISTER_MESSAGE)
    print(f"[+] Algorithms, scalers, encoders was sent to '{topic_name}' topic", flush=True)


def send_error_message(topic_name, model_id, info, exp):
    args = [model_id, type(exp).__name__, info + " " +
            str(exp), int(time.time())]
    service.send_message(topic_name, ERROR_MESSAGE(*args))
    print(f"    [+] Error was sent to '{topic_name}' topic", flush=True)


def parse_request(raw_request):
    try:
        req_json = json.loads(raw_request)
        return req_json
    except Exception as exp:
        print("    [x] Invalid Request", flush=True)
        print(f"    {type(exp).__name__}: {exp}", flush=True)
        return None


def check_label(label):
    print(f"    [!] Request type: {label}", flush=True)
    if label not in REQUEST_HANDLER_MAP:
        print(f"    [x] Invalid request type '{label}'", flush=True)
        return False
    return True


def handle_request(request, id, label):
    try:
        handler = REQUEST_HANDLER_MAP[label]()
        result = handler.handle(request)
        result['modelId'] = id
        result['modelLabel'] = label
        service.send_message(MainTopic.MAIN, result)
        print(
            f"    [+] Response was sent to '{MainTopic.MAIN}' topic", flush=True)
    except Exception as exp:
        print(f"    [x] {type(exp).__name__}: {exp}", flush=True)
        traceback.print_tb(exp.__traceback__)
        tb_err = ''.join(traceback.format_tb(exp.__traceback__))
        send_error_message(MainTopic.ERROR, id, f"Status: '{label}'", tb_err)


def on_message_received(channel, method, properties, request):
    print("[+] Received new message", flush=True)
    if (req := parse_request(request)) == None:
        return
    id, label = MainParser.parse(req).values()
    if check_label(label) == False:
        return
    handle_request(req, id, label)
    channel.basic_ack(method.delivery_tag)


if __name__ == '__main__':
    service = RabbitService(RabbitConfig.HOST, RabbitConfig.PORT,
                            RabbitConfig.USER, RabbitConfig.PASSWORD)
    service.add_exchange(MainTopic.MAIN, 'topic')
    service.add_exchange(MainTopic.ERROR, 'topic')
    service.add_exchange(ProviderConfig.TOPIC_NAME, 'topic')
    service.add_topic(MainTopic.MAIN)
    service.add_topic(MainTopic.ERROR)
    service.add_topic(ProviderConfig.TOPIC_NAME)
    send_register_message(MainTopic.MAIN)
    print(f"[*] Starting Consuming {ProviderConfig.TOPIC_NAME}", flush=True)
    service.start_consuming(ProviderConfig.TOPIC_NAME,
                            callback_func=on_message_received)
