import pandas as pd
from preprocessing import Preprocessor
from utils import get_values_by_shema as get_values, deserialize_object, parse_raw_params


class MainParser:
    @staticmethod
    def parse(request):
        id = get_values(request, [{'modelId': {"required": True}}])
        label = get_values(request, [{'modelLabel': {"required": True}}])
        return {"id": id, "label": label}


class CreateRequestParser:
    @staticmethod
    def parse(request):
        model_name = get_values(
            request, [{'model': {"required": True}}, {"name": {"required": True}}])
        model_params = get_values(request, [{'model': {"required": True}}, {
                                  "parameters": {"default": {}}}])
        return {"model_name": model_name, "model_params": parse_raw_params(model_params)}


class TrainRequestParser:
    @staticmethod
    def parse_features(request):
        feature_datas = get_values(request, [{'features': {"required": True}}])
        features_obj = get_values(
            request, [{'featuresHeader': {"required": True}}])
        feature_names = get_values(
            features_obj, [{'name': {"required": True}}])
        feature_types = get_values(
            features_obj, [{'type': {"required": True}}])
        features = pd.DataFrame(data=feature_datas, columns=feature_names)
        return features, feature_types

    @staticmethod
    def parse_scalers(request):
        scalers_obj = get_values(request, ['scalers'])
        if scalers_obj is not None:
            scaler_names = get_values(
                scalers_obj, [{"name": {"required": True}}])
            scaler_params = get_values(
                scalers_obj, [{"parameters": {"default": {}}}])
            return [Preprocessor.create(n, parse_raw_params(p)) for n, p in zip(
                scaler_names, scaler_params)]
        return None

    @staticmethod
    def parse_feature_encoders(request, feature_names, feature_types):
        feature_encoders = {}
        feature_encoders_obj = get_values(request, ['encodersFeatures'])
        if feature_encoders_obj is not None:
            encoders_feature_names = get_values(
                feature_encoders_obj, [{'featureName': {"required": True}}])
            feature_encoders_names = get_values(feature_encoders_obj, [
                                                {'encoder': {"required": True}}, {"name": {"required": True}}])
            feature_encoders_params = get_values(feature_encoders_obj, [
                                                 {'encoder': {"required": True}}, {"parameters": {"default": {}}}])
        for f_name, f_type in zip(feature_names, feature_types):
            if f_type == "numeric":
                continue
            if feature_encoders_obj is not None and f_name in encoders_feature_names:
                index = encoders_feature_names.index(f_name)
                _name = feature_encoders_names[index]
                _params = parse_raw_params(feature_encoders_params[index])
            else:
                _params = {}
                _name = "OrdinalEncoder"
                if f_type == "nominal":
                    _name = "OneHotEncoder"
            feature_encoders[f_name] = Preprocessor.create(_name, _params)
        return feature_encoders

    @staticmethod
    def parse_labels_encoder(request):
        labels_encoder_obj = get_values(request, ['encoderLabels'])
        if labels_encoder_obj is not None:
            labels_encoder_name = get_values(
                labels_encoder_obj, [{"name": {"required": True}}])
            labels_encoder_params = get_values(
                labels_encoder_obj, [{"parameters": {"default": {}}}])
            return Preprocessor.create(
                labels_encoder_name, parse_raw_params(labels_encoder_params))
        else:
            return Preprocessor.create("LabelEncoder", {})

    @staticmethod
    def parse(request):
        model = get_values(request, [{'model': {"required": True}}, {
                           "serializedData": {"required": True}}])
        model = deserialize_object(model)
        features, feature_types = TrainRequestParser.parse_features(request)
        labels = get_values(request, [{'labels': {"required": True}}])

        feature_encoders = TrainRequestParser.parse_feature_encoders(
            request, features.columns, feature_types)
        labels_encoder = TrainRequestParser.parse_labels_encoder(request)

        result = {
            "model": model, "features": features,
            "feature_types": feature_types, "labels": labels,
            "feature_encoders": feature_encoders, "labels_encoder": labels_encoder
        }
        scalers = TrainRequestParser.parse_scalers(request)
        if scalers is not None:
            result['scalers'] = scalers
        return result


class PredictRequestParser:
    @staticmethod
    def parse_features(request):
        feature_datas = get_values(request, [{'features': {"required": True}}])
        feature_names = get_values(
            request, [{'featuresHeader': {"required": True}}, {'name': {"required": True}}])
        return pd.DataFrame(data=feature_datas, columns=feature_names)

    @staticmethod
    def parse_scalers(request):
        scalers_obj = get_values(request, ['scalers'])
        if scalers_obj is not None:
            scaler_datas = get_values(
                scalers_obj, [{"serializedData": {"required": True}}])
            return [deserialize_object(s) for s in scaler_datas]
        return None

    @staticmethod
    def parse_feature_encoders(request):
        feature_encoders_obj = get_values(request, ['encodersFeatures'])
        if feature_encoders_obj is not None:
            encoders_feature_names = get_values(
                feature_encoders_obj, [{'featureName': {"required": True}}])
            feature_encoders_datas = get_values(feature_encoders_obj, [
                                                {'encoder': {"required": True}}, {"serializedData": {"required": True}}])
            return dict(zip(encoders_feature_names, list(map(deserialize_object, feature_encoders_datas))))
        return None

    @staticmethod
    def parse(request):
        model = get_values(request, [{'model': {"required": True}}, {
                           "serializedData": {"required": True}}])
        model = deserialize_object(model)
        features = PredictRequestParser.parse_features(request)

        result = {
            "model": model, "features": features,
            "has_scalers": False,
            "has_feature_encoders": False,
            "has_labels_encoder": False
        }

        scalers = PredictRequestParser.parse_scalers(request)
        if scalers is not None:
            result["scalers"] = scalers
            result["has_scalers"] = True

        feature_encoders = PredictRequestParser.parse_feature_encoders(request)
        if feature_encoders is not None:
            result["feature_encoders"] = feature_encoders
            result["has_feature_encoders"] = True

        labels_encoder_obj = get_values(request, ['encoderLabels'])
        if labels_encoder_obj is not None:
            labels_encoder_data = get_values(
                labels_encoder_obj, [{'serializedData': {"required": True}}])
            result["labels_encoder"] = deserialize_object(labels_encoder_data)
            result["has_labels_encoder"] = True

        return result
