import numpy as np
from classification import Classificator
from preprocessing import Preprocessor
from parsers import CreateRequestParser, TrainRequestParser, PredictRequestParser
from preprocessors import TrainRequestPreprocessor, PredictRequestPreprocessor
from packers import CreateRequestPacker, TrainRequestPacker, PredictRequestPacker


class CreateRequestHandler:
    def handle(self, request):
        self.state = CreateRequestParser.parse(request)
        self.state["model"] = Classificator.create(
            self.state["model_name"], self.state["model_params"])
        return CreateRequestPacker.pack(self.state)


class TrainRequestHandler:
    def handle(self, request):
        self.state = TrainRequestParser.parse(request)
        TrainRequestPreprocessor.preprocess(self.state)
        self.state["model"] = Classificator.train(
            self.state["model"], self.state["features"], self.state["labels"])
        return TrainRequestPacker.pack(self.state)


class PredictRequestHandler:
    def handle(self, request):
        self.state = PredictRequestParser.parse(request)
        PredictRequestPreprocessor.preprocess(self.state)
        self.state["predict_y"], self.state["predict_proba"] = Classificator.predict(
            self.state["model"], self.state["features"])
        self.state["predict_y"] = np.array(
            self.state["predict_y"]).reshape(-1, 1)
        if self.state["has_labels_encoder"]:
            self.state["predict_y"] = Preprocessor.inverse_transform(
            self.state["labels_encoder"], self.state["predict_y"])
        self.state["predict_y"] = self.state["predict_y"].tolist()
        return PredictRequestPacker.pack(self.state)
