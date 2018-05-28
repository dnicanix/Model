from sklearn.metrics import cohen_kappa_score as CohenKappaScore, confusion_matrix as ConfusionMatrix, \
                            hinge_loss as HingeLoss, matthews_corrcoef as MCC


class Evaluators:
    def __init__(self, test_labels, predicted_test_labels):
        self.test_labels = test_labels
        self.predicted_test_labels = predicted_test_labels


    def cohen_kappa_score(self):
        # Cohen’s kappa: a statistic that measures inter-annotator agreement.
        return CohenKappaScore(self.test_labels, self.predicted_test_labels)

    def confusion_matrix(self):
        # Compute confusion matrix to evaluate the accuracy of a classification
        return ConfusionMatrix(self.test_labels, self.predicted_test_labels)

    def hinge_loss(self):
        # Average hinge loss (non-regularized)
        return HingeLoss(self.test_labels, self.predicted_test_labels)

    def mcc(self):
        # Compute the Matthews correlation coefficient (MCC)
        return MCC(self.test_labels, self.predicted_test_labels)

