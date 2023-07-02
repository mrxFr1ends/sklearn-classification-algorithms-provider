from request_handlers import CreateRequestHandler, TrainRequestHandler, PredictRequestHandler
from classification import CLASSIFICATOR_ALGORITHMS
from preprocessing import SCALER_ALGORITHMS, ENCODER_ALGORITHMS


class RabbitConfig:
    # RabbitMQ Main Configuration
    HOST = 'rabbit_mq'
    PORT = 5672
    USER = 'guest'
    PASSWORD = 'guest'


class ProviderConfig:
    # RabbitMQ Provider Configuration
    NAME = 'sklearn_service'
    TOPIC_NAME = 'sklearn_service_topic'


class MainTopic:
    # RabbitMQ Main Topic Names
    MAIN = 'manager_in'
    ERROR = 'error-message'


REGISTER_MESSAGE = {
    "provider": ProviderConfig.NAME,
    "topic": ProviderConfig.TOPIC_NAME,
    "algorithms": CLASSIFICATOR_ALGORITHMS['algorithms'],
    "scalers": SCALER_ALGORITHMS['scalers'],
    "encoders": ENCODER_ALGORITHMS['encoders']
}


def ERROR_MESSAGE(model_id, errorType, errorMessage, datetime):
    return {
        "modelId": model_id,
        "errorType": errorType,
        "errorMessage": errorMessage,
        "localDateTime": datetime
    }


class RequestStatus:
    # Status values
    CREATE = 'CREATE'
    TRAIN = 'TRAIN'
    PREDICT = 'PREDICT'


REQUEST_HANDLER_MAP = {
    RequestStatus.CREATE: CreateRequestHandler,
    RequestStatus.TRAIN: TrainRequestHandler,
    RequestStatus.PREDICT: PredictRequestHandler
}
