from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
from preprocessing import Preprocessor


class TrainRequestPreprocessor:
    @staticmethod
    def preprocess(state):
        if len(state["scalers"]) > 0:
            pipe = make_pipeline(*state["scalers"])
            numeric_cols = state["features"].filter(items=[
                col for i, col in enumerate(state["features"].columns)
                if state["feature_types"][i] == 'numeric'
            ]).columns
            state["features"][numeric_cols] = pipe.fit_transform(
                state["features"][numeric_cols])

        for feature_name, encoder in state["feature_encoders"].items():
            encoded = Preprocessor.fit_transform(
                encoder, state["features"][feature_name].to_numpy().reshape(-1, 1))
            if len(encoded.shape) > 1 and encoded.shape[1] > 1:
                cols = [f'{feature_name}_{i}' for i in range(encoded.shape[1])]
                encoded_df = pd.DataFrame(encoded.toarray(), columns=cols)
                state["features"] = pd.concat(
                    [state["features"], encoded_df], axis=1).drop(feature_name, axis=1)
            else:
                state["features"][feature_name] = encoded
        state["features"] = state["features"].to_numpy().tolist()

        state["labels"] = Preprocessor.fit_transform(
            state["labels_encoder"], np.array(state["labels"]).reshape(-1, 1))


class PredictRequestPreprocessor:
    @staticmethod
    def preprocess(state):
        if state["has_scalers"]:
            pipe = make_pipeline(*state["scalers"])
            numeric_cols = pipe.feature_names_in_
            state["features"][numeric_cols] = pipe.transform(
                state["features"][numeric_cols])

        if state["has_feature_encoders"]:
            for feature_name, encoder in state["feature_encoders"].items():
                encoded = Preprocessor.transform(
                    encoder, state["features"][feature_name].to_numpy().reshape(-1, 1))
                if len(encoded.shape) > 1 and encoded.shape[1] > 1:
                    cols = [f'{feature_name}_{i}' for i in range(
                        encoded.shape[1])]
                    encoded_df = pd.DataFrame(encoded.toarray(), columns=cols)
                    state["features"] = pd.concat(
                        [state["features"], encoded_df], axis=1).drop(feature_name, axis=1)
                else:
                    state["features"][feature_name] = encoded
            state["features"] = state["features"].to_numpy().tolist()
