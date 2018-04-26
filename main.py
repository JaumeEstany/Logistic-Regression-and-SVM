import file_input as fi
import metrics as met
import logistic_regression as lr
import representation as rep
import svm
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():

    REPRESENT = True

    LOG = True
    SVM_linear = True
    SVM_poly = True
    SVM_rbf = True

    BASIC = True
    KFOLD = True
    LOO = True

    print '\n'

    x, y = fi.load_dataset()
    x, _ = fi.standarize(x, x)

    if REPRESENT:
        rep.represent_clfs(x, y)

    # ====================== Logistic regression ======================
    if LOG:
        print '====================== Logistic regression ======================'
        if BASIC:
            print "============== Holdout ==============="
            for part in range(5, 10):
                path = "results/Logistic/Holdout/"
                ratio = part/10.0

                x_train, y_train, x_val, y_val = fi.holdout(x, y, ratio)
                x_train, y_train, x_val = rep.reduce_dimensionsInto2(x_train, y_train, x_val)
                logireg, y_pred = lr.logisticRegression(x_train, y_train, x_val, y_val)

                precision, recall, F1Mes, accuracy, a, b, c = met.calculate_all_metrics(y_pred, y_val)
                print "TRAIN RATIO: ", ratio
                met.print_all_metrics(precision, recall, F1Mes, accuracy)

                path += str(part)
                met.generate_precision_recall_curve(logireg, x_val, y_val, path)
                met.generate_ROC_curve(logireg, x_val, y_val, path)



        if KFOLD:
            print "=============== K-fold ==============="
            Ks = [5, 10, 20, 30]

            for K in Ks:
                path = "results/Logistic/K-fold/"
                precision = 0
                recall = 0
                accuracy = 0
                pos_precision = 0
                pos_recall = 0

                for indexs_t, indexs_v in fi.kFold_indexs(x, K):
                    x_train, x_val, y_train, y_val = fi.kFold_dataset(x, y, indexs_t, indexs_v)

                    x_train, y_train, x_val = rep.reduce_dimensionsInto2(x_train, y_train, x_val)
                    logireg, y_pred = lr.logisticRegression(x_train, y_train, x_val, y_val)

                    prec, rec, F1, acc, pos_p, pos_r, pos_f1 = met.calculate_all_metrics(y_pred, y_val)
                    precision += prec
                    recall += rec
                    accuracy += acc
                    pos_precision += pos_p
                    pos_recall += pos_r

                print "K = ", K
                mean_precision = mean_recall = mean_f1score = 0
                if pos_precision > 0:
                    mean_precision = precision / float(pos_precision)
                if pos_recall > 0:
                    mean_recall = recall / float(pos_recall)
                if precision+recall > 0:
                    mean_f1score = (2*mean_precision*mean_recall)/(mean_precision+mean_recall)
                mean_accuracy = accuracy / float(K)
                met.print_all_metrics(mean_precision, mean_recall, mean_f1score, mean_accuracy)


        if LOO:
            print "============ Leave-One-Out ==========="
            N = len(y)
            print N, "total iterations"

            precision = 0
            recall = 0
            accuracy = 0
            pos_precision = 0
            pos_recall = 0

            for i in range(len(y)):
                Xtemp = np.hstack((x[:i, :].T, x[(i + 1):, :].T))
                Xtemp = Xtemp.T
                ytemp = np.hstack((y[:i], y[(i + 1):]))

                logireg, y_pred = lr.logisticRegression(Xtemp, ytemp, x[i, :], y[i])

                prec, rec, F1, acc, pos_p, pos_r, pos_f1 = met.calculate_all_metrics(y_pred, [y[i]])
                precision += prec
                recall += rec
                accuracy += acc
                pos_precision += pos_p
                pos_recall += pos_r

            mean_precision = mean_recall = mean_f1score = 0
            if pos_precision > 0:
                mean_precision = precision / float(pos_precision)
            if pos_recall > 0:
                mean_recall = recall / float(pos_recall)
            if precision + recall > 0:
                mean_f1score = (2 * mean_precision * mean_recall) / (mean_precision + mean_recall)
            mean_accuracy = accuracy / float(N)
            met.print_all_metrics(mean_precision, mean_recall, mean_f1score, mean_accuracy)


    # ========================= SVM linear ============================
    if SVM_linear:
        print "====================== SVM linear ========================"
        if BASIC:
            print "============== Holdout ==============="
            for part in range(5, 10):
                ratio = part/10.0
                x_train, y_train, x_val, y_val = fi.holdout(x, y, ratio)

                x_train, y_train, x_val = rep.reduce_dimensionsInto2(x_train, y_train, x_val)
                linear, y_pred = svm.linear_svm(x_train, y_train, x_val, y_val)

                errors = met.calculate_all_metrics(y_pred, y_val)
                print "TRAIN RATIO: ", ratio
                met.print_all_metrics(errors[0], errors[1], errors[2], errors[3])

                path = 'results/SVMlinear/Holdout/' + str(part)
                met.generate_precision_recall_curve(linear, x_val, y_val, path)
                met.generate_ROC_curve(linear, x_val, y_val, path)

        if KFOLD:
            print "=============== K-fold ==============="
            Ks = [5, 10, 20, 30]

            for K in Ks:
                precision = 0
                recall = 0
                accuracy = 0
                pos_precision = 0
                pos_recall = 0

                for indexs_t, indexs_v in fi.kFold_indexs(x, K):
                    x_train, x_val, y_train, y_val = fi.kFold_dataset(x, y, indexs_t, indexs_v)

                    x_train, y_train, x_val = rep.reduce_dimensionsInto2(x_train, y_train, x_val)
                    linear, y_pred = svm.linear_svm(x_train, y_train, x_val, y_val)

                    prec, rec, F1, acc, pos_p, pos_r, pos_f1 = met.calculate_all_metrics(y_pred, y_val)
                    precision += prec
                    recall += rec
                    accuracy += acc
                    pos_precision += pos_p
                    pos_recall += pos_r

                print "K = ", K
                mean_precision = mean_recall = mean_f1score = 0
                if pos_precision > 0:
                    mean_precision = precision / float(pos_precision)
                if pos_recall > 0:
                    mean_recall = recall / float(pos_recall)
                if precision+recall > 0:
                    mean_f1score = (2*mean_precision*mean_recall)/(mean_precision+mean_recall)
                mean_accuracy = accuracy / float(K)
                met.print_all_metrics(mean_precision, mean_recall, mean_f1score, mean_accuracy)

        if LOO:
            print "============ Leave-One-Out ==========="
            N = len(y)
            print N, "total iterations"

            precision = 0
            recall = 0
            accuracy = 0
            pos_precision = 0
            pos_recall = 0

            for i in range(len(y)):
                Xtemp = np.hstack((x[:i, :].T, x[(i + 1):, :].T))
                Xtemp = Xtemp.T
                ytemp = np.hstack((y[:i], y[(i + 1):]))

                linear, y_pred = svm.linear_svm(Xtemp, ytemp, x[i, :], y[i])

                prec, rec, F1, acc, pos_p, pos_r, pos_f1 = met.calculate_all_metrics(y_pred, [y[i]])
                precision += prec
                recall += rec
                accuracy += acc
                pos_precision += pos_p
                pos_recall += pos_r

            mean_precision = mean_recall = mean_f1score = 0
            if pos_precision > 0:
                mean_precision = precision / float(pos_precision)
            if pos_recall > 0:
                mean_recall = recall / float(pos_recall)
            if precision + recall > 0:
                mean_f1score = (2 * mean_precision * mean_recall) / (mean_precision + mean_recall)
            mean_accuracy = accuracy / float(N)
            met.print_all_metrics(mean_precision, mean_recall, mean_f1score, mean_accuracy)

    # ========================== SVM poly =============================
    if SVM_poly:
        print "====================== SVM poly =========================="
        if BASIC:
            print "============== Holdout ==============="
            for part in range(5, 10):
                ratio = part/10.0
                for degree in range(2, 5):
                    print "Degree = " + str(degree)
                    print "TRAIN RATIO: ", ratio
                    x_train, y_train, x_val, y_val = fi.holdout(x, y, ratio)

                    x_train, y_train, x_val = rep.reduce_dimensionsInto2(x_train, y_train, x_val)
                    svmobj, y_pred = svm.poly_svm(x_train, y_train, x_val, y_val, degree)

                    errors = met.calculate_all_metrics(y_pred, y_val)
                    met.print_all_metrics(errors[0], errors[1], errors[2], errors[3])

                    path = 'results/SVMpoly/Holdout/' + str(part) + 'D=' + str(degree)
                    met.generate_precision_recall_curve(svmobj, x_val, y_val, path)
                    met.generate_ROC_curve(svmobj, x_val, y_val, path)


        if KFOLD:
            print "=============== K-fold ==============="
            Ks = [5, 10, 20, 30]

            for K in Ks:
                precision = 0
                recall = 0
                accuracy = 0
                pos_precision = 0
                pos_recall = 0

                for indexs_t, indexs_v in fi.kFold_indexs(x, K):
                    x_train, x_val, y_train, y_val = fi.kFold_dataset(x, y, indexs_t, indexs_v)

                    x_train, y_train, x_val = rep.reduce_dimensionsInto2(x_train, y_train, x_val)
                    svmobj, y_pred = svm.poly_svm(x_train, y_train, x_val, y_val, 2)

                    prec, rec, F1, acc, pos_p, pos_r, pos_f1 = met.calculate_all_metrics(y_pred, y_val)
                    precision += prec
                    recall += rec
                    accuracy += acc
                    pos_precision += pos_p
                    pos_recall += pos_r

                print "K = ", K
                mean_precision = mean_recall = mean_f1score = 0
                if pos_precision > 0:
                    mean_precision = precision / float(pos_precision)
                if pos_recall > 0:
                    mean_recall = recall / float(pos_recall)
                if precision+recall > 0:
                    mean_f1score = (2*mean_precision*mean_recall)/(mean_precision+mean_recall)
                mean_accuracy = accuracy / float(K)
                met.print_all_metrics(mean_precision, mean_recall, mean_f1score, mean_accuracy)

        if LOO:
            print "============ Leave-One-Out ==========="
            N = len(y)
            print N, "total iterations"

            precision = 0
            recall = 0
            accuracy = 0
            pos_precision = 0
            pos_recall = 0

            for i in range(len(y)):
                Xtemp = np.hstack((x[:i, :].T, x[(i + 1):, :].T))
                Xtemp = Xtemp.T
                ytemp = np.hstack((y[:i], y[(i + 1):]))

                poly, y_pred = svm.poly_svm(Xtemp, ytemp, x[i, :], y[i])

                prec, rec, F1, acc, pos_p, pos_r, pos_f1 = met.calculate_all_metrics(y_pred, [y[i]])
                precision += prec
                recall += rec
                accuracy += acc
                pos_precision += pos_p
                pos_recall += pos_r

            mean_precision = mean_recall = mean_f1score = 0
            if pos_precision > 0:
                mean_precision = precision / float(pos_precision)
            if pos_recall > 0:
                mean_recall = recall / float(pos_recall)
            if precision + recall > 0:
                mean_f1score = (2 * mean_precision * mean_recall) / (mean_precision + mean_recall)
            mean_accuracy = accuracy / float(N)
            met.print_all_metrics(mean_precision, mean_recall, mean_f1score, mean_accuracy)

    # ========================== SVM rbf ==============================
    if SVM_rbf:
        print "====================== SVM rbf ==========================="
        if BASIC:
            print "============== Holdout ==============="
            for part in range(5, 10):
                ratio = part / 10.0
                x_train, y_train, x_val, y_val = fi.holdout(x, y, ratio)

                x_train, y_train, x_val = rep.reduce_dimensionsInto2(x_train, y_train, x_val)
                svmobj, y_pred = svm.rbf_svm(x_train, y_train, x_val, y_val)

                errors = met.calculate_all_metrics(y_pred, y_val)
                print "TRAIN RATIO: ", ratio
                met.print_all_metrics(errors[0], errors[1], errors[2], errors[3])

                path = 'results/SVMgauss/Holdout/' + str(part)
                met.generate_precision_recall_curve(svmobj, x_val, y_val, path)
                met.generate_ROC_curve(svmobj, x_val, y_val, path)

        if KFOLD:
            print "=============== K-fold ==============="
            Ks = [5, 10, 20, 30]

            for K in Ks:
                precision = 0
                recall = 0
                accuracy = 0
                pos_precision = 0
                pos_recall = 0

                for indexs_t, indexs_v in fi.kFold_indexs(x, K):
                    x_train, x_val, y_train, y_val = fi.kFold_dataset(x, y, indexs_t, indexs_v)

                    x_train, y_train, x_val = rep.reduce_dimensionsInto2(x_train, y_train, x_val)
                    svmobj, y_pred = svm.rbf_svm(x_train, y_train, x_val, y_val)

                    prec, rec, F1, acc, pos_p, pos_r, pos_f1 = met.calculate_all_metrics(y_pred, y_val)
                    precision += prec
                    recall += rec
                    accuracy += acc
                    pos_precision += pos_p
                    pos_recall += pos_r

                print "K = ", K
                mean_precision = mean_recall = mean_f1score = 0
                if pos_precision > 0:
                    mean_precision = precision / float(pos_precision)
                if pos_recall > 0:
                    mean_recall = recall / float(pos_recall)
                if precision+recall > 0:
                    mean_f1score = (2*mean_precision*mean_recall)/(mean_precision+mean_recall)
                mean_accuracy = accuracy / float(K)
                met.print_all_metrics(mean_precision, mean_recall, mean_f1score, mean_accuracy)

        if LOO:
            print "============ Leave-One-Out ==========="
            N = len(y)
            print N, "total iterations"

            precision = 0
            recall = 0
            accuracy = 0
            pos_precision = 0
            pos_recall = 0

            for i in range(len(y)):
                Xtemp = np.hstack((x[:i, :].T, x[(i + 1):, :].T))
                Xtemp = Xtemp.T
                ytemp = np.hstack((y[:i], y[(i + 1):]))

                rbf, y_pred = svm.rbf_svm(Xtemp, ytemp, x[i, :], y[i])

                prec, rec, F1, acc, pos_p, pos_r, pos_f1 = met.calculate_all_metrics(y_pred, [y[i]])
                precision += prec
                recall += rec
                accuracy += acc
                pos_precision += pos_p
                pos_recall += pos_r

            mean_precision = mean_recall = mean_f1score = 0
            if pos_precision > 0:
                mean_precision = precision / float(pos_precision)
            if pos_recall > 0:
                mean_recall = recall / float(pos_recall)
            if precision + recall > 0:
                mean_f1score = (2 * mean_precision * mean_recall) / (mean_precision + mean_recall)
            mean_accuracy = accuracy / float(N)
            met.print_all_metrics(mean_precision, mean_recall, mean_f1score, mean_accuracy)





if __name__ == '__main__':
    main()
