from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, PassiveAggressiveClassifier, Perceptron, RidgeClassifier, RidgeClassifierCV, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier

from classification.classificator_algorithms import CLASSIFICATOR_ALGORITHMS


class Classificator:
    # @staticmethod
    # def get_algorithms():
    #     return [class_.__name__ for _, class_ in all_estimators(type_filter='classifier')]

    # @staticmethod
    # def get_hyperparams(model_name):
    #     return globals()[model_name]._parameter_constraints.keys()

    @staticmethod
    def create(model_name, params):
        return globals()[model_name](**params)

    @staticmethod
    def train(model, x, y):
        return model.fit(x, y)

    # @staticmethod
    # def train(model, x, y):
    #     clf = model.fit(x, y)
    #     fi = getattr(clf, 'feature_importances_', None)
    #     return clf, fi.tolist() if fi is not None else None

    @staticmethod
    def predict(model, x):
        predict_y = model.predict(x).tolist()
        predict_proba = model.predict_proba(x).tolist(
        ) if hasattr(model, 'predict_proba') else None
        return predict_y, predict_proba
