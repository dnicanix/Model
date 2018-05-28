import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.externals import joblib

class Classififers:
    def __init__(self, train_texts_tfidf, train_labels):
        self.train_texts_tfidf = train_texts_tfidf
        self.train_labels = train_labels
        self.gbc_filename = 'gbc_ngrams3_model.pkl'
        self.gpc_filename = 'gpc_ngrams3_model.pkl'
        self.lsvc_filename = 'lsvc_ngrams3_model.pkl'
        self.lr_filename = 'lr_ngrams3_model.pkl'
        self.lrcv_filename = 'lrcv_ngrams3_model.pkl'
        self.sgdc_filename = 'sgdc_ngrams3_model.pkl'
        self.p_filename = 'p_ngrams3_model.pkl'
        self.pac_filename = 'pac_ngrams3_model.pkl'

    def gradient_boosting_classifier(self):
        model = OneVsRestClassifier(GradientBoostingClassifier()).fit(self.train_texts_tfidf, self.train_labels)
        self.save_model(model, self.gbc_filename)
        return model

    def gaussian_process_classifier(self):
        model = OneVsRestClassifier(GaussianProcessClassifier()).fit(self.train_texts_tfidf, self.train_labels)
        self.save_model(model, self.gpc_filename)
        return model

    def linear_svc(self):
        model = OneVsRestClassifier(LinearSVC()).fit(self.train_texts_tfidf, self.train_labels)
        self.save_model(model, self.lsvc_filename)
        return model

    def logistic_regression(self):
        model = OneVsRestClassifier(LogisticRegression()).fit(self.train_texts_tfidf, self.train_labels)
        self.save_model(model, self.lr_filename)
        return model

    def logistic_regression_cv(self):
        model = OneVsRestClassifier(LogisticRegressionCV()).fit(self.train_texts_tfidf, self.train_labels)
        self.save_model(model, self.lrcv_filename)
        return model

    def sgdc_classifier(self):
        model = OneVsRestClassifier(SGDClassifier(max_iter=1000)).fit(self.train_texts_tfidf, self.train_labels)
        self.save_model(model, self.sgdc_filename)
        return model

    def perceptron(self):
        model = OneVsRestClassifier(Perceptron(max_iter=1000)).fit(self.train_texts_tfidf, self.train_labels)
        self.save_model(model, self.p_filename)
        return model

    def passive_agressive_classifer(self):
        model = OneVsRestClassifier(PassiveAggressiveClassifier(max_iter=1000)).fit(self.train_texts_tfidf, self.train_labels)
        self.save_model(model, self.pac_filename)
        return model

    def save_model(self, model, filename):
        joblib.dump(model, filename)