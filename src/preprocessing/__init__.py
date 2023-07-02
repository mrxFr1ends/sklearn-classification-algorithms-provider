from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from preprocessing.scaler_algorithms import SCALER_ALGORITHMS
from preprocessing.encoder_algorithms import ENCODER_ALGORITHMS


class Preprocessor:
    @staticmethod
    def create(class_name, parameters):
        return globals()[class_name](**parameters)

    @staticmethod
    def fit_transform(class_object, X):
        return class_object.fit_transform(X)

    @staticmethod
    def transform(class_object, X):
        return class_object.transform(X)

    @staticmethod
    def inverse_transform(class_object, X):
        return class_object.inverse_transform(X)
