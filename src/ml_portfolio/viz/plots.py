import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay

def plot_roc(estimator, X_test, y_test):
    RocCurveDisplay.from_estimator(estimator, X_test, y_test)
    plt.title("ROC Curve")
    plt.show()

def plot_pr(estimator, X_test, y_test):
    PrecisionRecallDisplay.from_estimator(estimator, X_test, y_test)
    plt.title("Precision-Recall Curve")
    plt.show()

def plot_cm(estimator, X_test, y_test):
    ConfusionMatrixDisplay.from_estimator(estimator, X_test, y_test)
    plt.title("Confusion Matrix")
    plt.show()
