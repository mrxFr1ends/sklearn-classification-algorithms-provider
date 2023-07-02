from utils import serialize_object


class CreateRequestPacker:
    @staticmethod
    def pack(state):
        response = {
            "model": {
                "serializedData": serialize_object(state["model"])
            }
        }
        return response


class TrainRequestPacker:
    @staticmethod
    def pack(state):
        response = {
            "model": {
                "serializedData": serialize_object(state["model"])
            }
        }
        if len(state["scalers"]) > 0:
            response["scalers"] = [{
                "serializedData": s
            } for s in [serialize_object(s) for s in state["scalers"]]]

        response["encodersFeatures"] = [{
            "encoder": {"serializedData": e}
        } for e in [serialize_object(e) for e in state["feature_encoders"].values()]]

        response["encoderLabels"] = {}
        response["encoderLabels"]["serializedData"] = serialize_object(
            state["labels_encoder"])
        return response


class PredictRequestPacker:
    @staticmethod
    def pack(state):
        response = {
            "labels": state["predict_y"]
        }
        if state["predict_proba"] is not None:
            response["distributions"] = state["predict_proba"]
        return response
