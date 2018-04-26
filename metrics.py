
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


def calculate_all_metrics(y_predicted, y_true):

    accuracy = metrics.accuracy_score(y_true, y_predicted)

    precision = 0.0
    recall = 0.0
    f1score = 0.0

    valid_precision = (1 in y_predicted)
    valid_recall = (1 in y_true)
    valid_f1 = (valid_precision and valid_recall)

    if valid_precision:
        precision = metrics.precision_score(y_true, y_predicted)

    if valid_recall:
        recall = metrics.recall_score(y_true, y_predicted)

    if valid_f1:
        f1score = metrics.f1_score(y_true, y_predicted)

    return precision, recall, f1score, accuracy, valid_precision, valid_precision, valid_f1


def print_all_metrics(precision, recall, F1score, accuracy):
    print "Precision: ", precision
    print "Recall: ", recall
    print "F1-Score: ", F1score
    print "Accuracy: ", accuracy
    print "\n"


def generate_precision_recall_curve(model, x_val, y_val, path):

    probs = model.predict_proba(x_val)

    plt.figure()
    precision, recall, a = metrics.precision_recall_curve(y_val == 1, probs[:, 1])
    average_precision = metrics.average_precision_score(y_val == 1, probs[:, 1])

    plt.plot(recall, precision, label=("Precision - Recall curve : " + str(average_precision)))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="upper right")

    final_path = path + "PrecRec"
    plt.savefig(final_path)
    plt.close()



def generate_ROC_curve(model, x_val, y_val, path):
    probs = model.predict_proba(x_val)

    plt.figure()
    fpr, tpr, a = metrics.roc_curve(y_val == 1, probs[:, 1])
    roc_auc = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, label=("ROC curve : " + str(roc_auc)))
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="upper right")

    final_path = path + "ROC"
    plt.savefig(final_path)
    plt.close()
