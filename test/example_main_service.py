import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../src'))
import warnings
warnings.filterwarnings('ignore')
import traceback
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import NotFittedError
import json
import pandas as pd
import numpy as np
from utils import deserialize_object, serialize_object
from rabbit_service import RabbitService
from config import REGISTER_MESSAGE, RabbitConfig, MainTopic, RequestStatus, ProviderConfig
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder

MODEL_ID = 'test_model_id'
X = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
Y = [19, 20, 30]
test_data = {}

def received_object(body):
    print(f"{' ' * 8}[!] Received {len(body)} bytes")
    try: 
        return json.loads(body)
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return None

def is_fitted(clf, test_data):
    try: clf.predict(test_data)
    except NotFittedError:
        return False
    return True

def try_deserialize(object):
    try:
        print(f"{' ' * 8}[!] Try deserialize object")
        return deserialize_object(object)
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return None


def register_test():
    print("[!] Register provider test")
    print("    [*] Waiting register request")
    message = service.consume_message(MainTopic.MAIN)
    assert (request := received_object(message)) is not None, 'Error when formatting in json'
    assert len(set(REGISTER_MESSAGE) & set(request)) == len(REGISTER_MESSAGE), "Incorrect request to register"
    assert request['provider'] == REGISTER_MESSAGE['provider'], f"Incorrect provider name {request['provider']}"
    assert request['topic'] == REGISTER_MESSAGE['topic'], f"Incorrect topic name {request['topic']}"
    assert request['algorithms'] == REGISTER_MESSAGE['algorithms'], f"Incorrect algorithms"
    assert request['scalers'] == REGISTER_MESSAGE['scalers'], f"Incorrect scalers"
    assert request['encoders'] == REGISTER_MESSAGE['encoders'], f"Incorrect encoders"


def create_model_test():
    CREATE_MESSAGE = {
        "modelLabel": RequestStatus.CREATE,
        "modelId": MODEL_ID,
        "model": {
            "name": "ExtraTreesClassifier",
            "parameters": [
                {"name": "n_estimators", "value": "50"},
                {"name": "criterion", "value": "gini"},
                {"name": "random_state", "value": "0"}
            ]
        }
    }

    print("[!] Create model test")
    print(f"    [!] Send '{RequestStatus.CREATE}' message")
    service.send_message(ProviderConfig.TOPIC_NAME, CREATE_MESSAGE)
    test_data['valid_model'] = ExtraTreesClassifier(n_estimators=50, criterion="gini", random_state=0)

    print(f"    [*] Waiting '{RequestStatus.CREATE}' response")
    message = service.consume_message(MainTopic.MAIN)
    assert (response := received_object(message)) is not None, 'Error when formatting in json'
    assert 'modelId' in response and response['modelId'] == MODEL_ID, 'Missing or incorrect "modelId"'
    assert 'model' in response and 'serializedData' in response['model'], 'Missing "model" or "serializedData"'
    assert (model := try_deserialize(response['model']['serializedData'])) is not None, 'Error when deserialized model' 

    assert type(model) == type(test_data["valid_model"]), f'Incorrect type of classificator: {type(model)}'
    assert model.get_params() == test_data["valid_model"].get_params(), f'Incorrect params: {model.get_params()}'
    assert is_fitted(model, [X[0]]) == False, 'Model was fitted'
    test_data['created_model'] = model


def train_model_test():
    TRAIN_MESSAGE = {
        "modelLabel": RequestStatus.TRAIN,
        "modelId": MODEL_ID,
        "features": X,
        "labels": Y,
        "featuresHeader": [
            {"name": "feature1", "type": "numeric"}, 
            {"name": "feature2", "type": "nominal"}, 
            {"name": "feature3", "type": "ordinal"}, 
            {"name": "feature4", "type": "ordinal"}
        ],
        "model": { "serializedData": None },
        "scalers": [{
            "name": "StandardScaler"
        }],
        "encodersFeatures": [{
            "featureName": "feature2",
            "encoder": { "name": "LabelEncoder" }
        }, {
            "featureName": "feature3",
            "encoder": {
                "name": "OneHotEncoder",
                "parameters": [{"name": "drop", "value": "first"}]
            }
        }],
        "encoderLabels": {
            "name": "OrdinalEncoder"
        }
    }

    def _compare_dicts(left: dict, right: dict):
        valid_params = left.copy()
        params = right.copy()
        if 'encoded_missing_value' in valid_params and 'encoded_missing_value' in params:
            del valid_params['encoded_missing_value']
            del params['encoded_missing_value']
        return valid_params == params

    print("[!] Train model test")
    TRAIN_MESSAGE['model']['serializedData'] = serialize_object(test_data['created_model'])

    print(f"    [!] Send '{RequestStatus.TRAIN}' message")
    service.send_message(ProviderConfig.TOPIC_NAME, TRAIN_MESSAGE)
    test_data['valid_scaler'] = StandardScaler()
    test_data['valid_feature_encoders'] = [LabelEncoder(), OneHotEncoder(drop="first"), OrdinalEncoder()]
    test_data['valid_labels_encoder'] = OrdinalEncoder()

    prepared_X_df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4'])
    prepared_X_df['feature1'] = test_data['valid_scaler'].fit_transform(prepared_X_df[['feature1']])
    prepared_X_df['feature2'] = test_data['valid_feature_encoders'][0].fit_transform(prepared_X_df[['feature2']])
    prepared_X_df[['feature3_0', 'feature3_1']] = test_data['valid_feature_encoders'][1].fit_transform(prepared_X_df[['feature3']]).toarray()
    prepared_X_df = prepared_X_df.drop(['feature3'], axis=1)
    prepared_X_df[['feature4']] = test_data['valid_feature_encoders'][2].fit_transform(prepared_X_df[['feature2']])
    prepared_Y = test_data['valid_labels_encoder'].fit_transform(np.array(Y).reshape(-1, 1))
    test_data['valid_prepared_X_df'] = prepared_X_df
    test_data['valid_prepared_Y'] = prepared_Y
    test_data['valid_model'] = test_data['valid_model'].fit(prepared_X_df.to_numpy().tolist(), prepared_Y)
    
    print(f"    [*] Waiting '{RequestStatus.TRAIN}' response")
    message = service.consume_message(MainTopic.MAIN)
    assert (response := received_object(message)) is not None, 'Error when formatting in json'
    assert 'modelId' in response and response['modelId'] == MODEL_ID, 'Missing or incorrect "modelId"'
    assert 'model' in response and 'serializedData' in response['model'], 'Missing "model" or "serializedData"'
    assert (model := try_deserialize(response['model']['serializedData'])) is not None, 'Error when deserialized model'
    assert type(model) == type(test_data['valid_model']), f'Incorrect type of classificator: {type(model)}'
    assert model.get_params() == test_data['valid_model'].get_params(), f'Incorrect params: {model.get_params()}'
    assert is_fitted(model, [test_data['valid_prepared_X_df'].to_numpy().tolist()[0]]) == True, 'Model was not fitted'
    assert (test_data['valid_model'].classes_ == model.classes_).all(), f'The classes labels not equal: {model.classes_}'
    test_data['trained_model'] = model

    assert 'scalers' in response and len(response['scalers']) == 1, 'Missing "scalers" or len != 1'
    assert (scaler := try_deserialize(response['scalers'][0]['serializedData'])) is not None and \
           type(scaler) == type(test_data['valid_scaler']), 'Inccorect scaler'
    assert _compare_dicts(scaler.get_params(), test_data['valid_scaler'].get_params()), f'Incorrect params: {scaler.get_params()}'
    test_data['created_scaler'] = scaler
    
    assert 'encodersFeatures' in response and \
           len(response['encodersFeatures']) == len(test_data['valid_feature_encoders']), \
           f'Missing "encodersFeatures" or {len(response["encodersFeatures"])} != {len(test_data["valid_feature_encoders"])}'
    feature_encoders = [try_deserialize(e['encoder']['serializedData']) for e in response['encodersFeatures']]
    assert None not in feature_encoders, 'Incorrect feature encoder(-s)'
    for i in range(len(test_data['valid_feature_encoders'])):
        assert type(feature_encoders[i]) ==  type(test_data['valid_feature_encoders'][i]), f'Incorrect type {i+1} encoder: {type(feature_encoders[i])}'
        assert _compare_dicts(feature_encoders[i].get_params(), test_data['valid_feature_encoders'][i].get_params()), \
            f'Incorrect params: {feature_encoders[i].get_params()}'
    test_data['created_feature_encoders'] = feature_encoders

    assert 'encoderLabels' in response and \
           (labels_encoder := try_deserialize(response['encoderLabels']['serializedData'])) is not None and \
           type(labels_encoder) == type(test_data['valid_labels_encoder']), 'Inccorect label encoder'
    assert type(labels_encoder) ==  type(test_data['valid_labels_encoder']), f'Incorrect type labels encoder: {type(labels_encoder)}'
    assert _compare_dicts(labels_encoder.get_params(), test_data['valid_labels_encoder'].get_params()), \
        f'Incorrect params: {labels_encoder.get_params()}'
    test_data['created_labels_encoder'] = labels_encoder


def model_predict_test():
    PREDICT_MESSAGE = {
        "modelLabel": RequestStatus.PREDICT,
        "modelId": MODEL_ID,
        "model": { "serializedData": None },
        "features": X,
        "featuresHeader": [
            {"name": "feature1"}, 
            {"name": "feature2"}, 
            {"name": "feature3"}, 
            {"name": "feature4"}],
        "scalers": [{ "serializedData": None }],
        "encodersFeatures": [{
            "featureName": "feature2",
            "encoder": { "serializedData": None }
        }, {
            "featureName": "feature3",
            "encoder": { "serializedData": None }
        }, {
            "featureName": "feature4",
            "encoder": { "serializedData": None }
        }],
        "encoderLabels": { "serializedData": None }
    }

    print("[!] Model predict test")
    PREDICT_MESSAGE['model']['serializedData'] = serialize_object(test_data['trained_model'])
    PREDICT_MESSAGE['scalers'][0]['serializedData'] = serialize_object(test_data['created_scaler'])
    PREDICT_MESSAGE['encodersFeatures'][0]['encoder']['serializedData'] = serialize_object(test_data['created_feature_encoders'][0])
    PREDICT_MESSAGE['encodersFeatures'][1]['encoder']['serializedData'] = serialize_object(test_data['created_feature_encoders'][1])
    PREDICT_MESSAGE['encodersFeatures'][2]['encoder']['serializedData'] = serialize_object(test_data['created_feature_encoders'][2])
    PREDICT_MESSAGE["encoderLabels"]['serializedData'] = serialize_object(test_data['created_labels_encoder'])

    print(f"    [!] Send '{RequestStatus.PREDICT}' message")
    service.send_message(ProviderConfig.TOPIC_NAME, PREDICT_MESSAGE)
    test_data["valid_y_predict"] = test_data["valid_model"].predict(test_data['valid_prepared_X_df'].to_numpy().tolist()).reshape(-1, 1)
    test_data["valid_y_predict"] = test_data["valid_labels_encoder"].inverse_transform(test_data["valid_y_predict"])
    test_data["valid_y_proba"] = test_data["valid_model"].predict_proba(test_data['valid_prepared_X_df'].to_numpy().tolist())

    print(f"    [*] Waiting '{RequestStatus.PREDICT}' response")
    message = service.consume_message(MainTopic.MAIN)
    assert (response := received_object(message)), 'Error when formatting in json'
    assert 'modelId' in response and response['modelId'] == MODEL_ID, 'Missing or incorrect "modelId"'
    assert 'labels' in response, 'Missing "labels"'
    assert (response['labels'] == test_data['valid_y_predict']).all(), \
        f'Incorrect predict classes\n{response["labels"]}\nnot equal\n{test_data["valid_y_predict"]}'
    assert 'distributions' in response, 'Missing "distributions"'
    assert (response['distributions'] == test_data['valid_y_proba']).all(), \
        f'Incorrect proba classes\n{response["distributions"]}\nnot equal\n{test_data["valid_y_proba"]}'


if __name__ == "__main__":
    service = RabbitService('127.0.0.1', RabbitConfig.PORT,
                            RabbitConfig.USER, RabbitConfig.PASSWORD)
    
    tests = [register_test, create_model_test, train_model_test, model_predict_test]
    for test in tests:
        test()
        print(f'    [+] Test passed')