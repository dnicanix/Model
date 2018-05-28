from sklearn.model_selection import cross_validate,cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier, Perceptron, PassiveAggressiveClassifier

class CrossValidation:
    def __init__(self, train_texts_tfidf, train_label):
        self.train_texts_tfidf = train_texts_tfidf
        self.train_label = train_label
        self.cv = 10
        self.scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

    def performCV(self):
        print("Performance for GBC: ")
        scores = cross_validate(GradientBoostingClassifier(), self.train_texts_tfidf, self.train_label, cv=self.cv, scoring=self.scoring)
        self.getPerformance(scores)

        print("Performance for LSVC: ")
        scores = cross_validate(LinearSVC(), self.train_texts_tfidf, self.train_label, cv=self.cv, scoring=self.scoring)
        self.getPerformance(scores)

        print("Performance for LR: ")
        scores = cross_validate(LogisticRegression(), self.train_texts_tfidf, self.train_label, cv=self.cv, scoring=self.scoring)
        self.getPerformance(scores)

        print("Performance for LRCV: ")
        scores = cross_validate(LogisticRegressionCV(), self.train_texts_tfidf, self.train_label, cv=self.cv, scoring=self.scoring)
        self.getPerformance(scores)

        print("Performance for SGDC: ")
        scores = cross_validate(SGDClassifier(), self.train_texts_tfidf, self.train_label, cv=self.cv, scoring=self.scoring)
        self.getPerformance(scores)

        print("Performance for P: ")
        scores = cross_validate(Perceptron(), self.train_texts_tfidf, self.train_label, cv=self.cv, scoring=self.scoring)
        self.getPerformance(scores)

        print("Performance for PAC: ")
        scores = cross_validate(PassiveAggressiveClassifier(), self.train_texts_tfidf, self.train_label, cv=self.cv, scoring=self.scoring)
        self.getPerformance(scores)

    def getPerformance(self, scores):
        print('Accuracy', scores['test_accuracy'].mean())
        print('Precision', scores['test_precision_weighted'].mean())
        print('Recall', scores['test_recall_weighted'].mean())
        print('F1 Score', scores['test_f1_weighted'].mean())