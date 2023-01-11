import src.load_data as ld
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical
import numpy as np
import time
import src.model as md

def svm():
    x, y = ld.load_data('data/A_Z Handwritten Data.csv')
    kf = KFold(n_splits=5, shuffle=True)
    kernels = ['rbf', 'polynomial', 'linear']
    Cs = [1, 0.1]
    log = []

    for k in kernels:
        for C in Cs:
            name = 'SVM_' + k + '_' + str(C)
            print(name)
            start = time.time()
            results = []
            for train_i, test_i in kf.split(x, y):
                print('fold 1')
                print('======================================================')
                train_x, test_x = x[train_i], x[test_i]
                train_y, test_y = y[train_i], y[test_i]

                svm = md.SVM(name, C, k)
                svm.train(train_x, train_y)

                results.append(svm.evaluate(test_x, test_y))

            end = time.time()
            log.append(name)
            log.append(np.mean(results))
            log.append((end-start)/5)
            print(name)
            print('Accuracy: %.2f%%' % (np.mean(results)*100))
            print('Average Training Time:' + str((end-start)/5))

    fp = open('SVM_results.txt', mode='w')
    fp.write(str(log))

def build_mlp(name, a, d, o):
    mlp = md.MLP(name)
    mlp.build(act=a, dropout_rate=d)
    mlp.compile(opt=o)
    return mlp

def mlp():
    x, y = ld.load_data('data/A_Z Handwritten Data.csv')
    y = to_categorical(y)
    kf = KFold(n_splits=5, shuffle=True)
    opts = ['sgd', 'adam', 'rmsprop']
    acts = ['sigmoid', 'relu']
    drops = [0, 0.2]
    log = []

    for o in opts:
        for a in acts:
            for d in drops:
                name = 'MLP_' + o + '_' + a + '_' + str(d)
                start = time.time()
                mlp = KerasClassifier(build_fn=build_mlp(name, a, d, o), epochs=10, batch_size=256)
                results = cross_val_score(mlp, x, y, cv=kf)
                end = time.time()
                log.append(name)
                log.append(results.mean())
                log.append((end-start)/5)
                print(name)
                print('Accuracy: %.2f%% (%.2f)' % (results.mean()*100, results.std()))
                print('Average Training Time:' + str((end-start)/5))

    fp = open('MLP_results.txt', mode='w')
    fp.write(str(log))
    fp.close()

def build_cnn(name):
    cnn = md.CNN(name)
    cnn.build()
    cnn.compile(opt='adam')
    return cnn

def cnn():
    x, y = ld.load_data('data/A_Z Handwritten Data.csv')
    y = to_categorical(y)
    kf = KFold(n_splits=5, shuffle=True)
    batch_size = [128, 256, 512]
    epochs = [5, 10, 20]
    log = []

    for b in batch_size:
        for e in epochs:
            name = 'CNN_' + str(b) + '_' + str(e)
            start = time.time()
            results = []
            for train_i, test_i in kf.split(x, y):
                train_x, test_x = x[train_i], x[test_i]
                train_y, test_y = y[train_i], y[test_i]

                cnn = build_cnn(name)
                cnn.train(train_x, train_y, b_size=b, ep=e)
                
                results.append(md.evaluate(name, 'cnn', test_x, test_y)[1])

            end = time.time()
            log.append(name)
            log.append(np.mean(results))
            log.append((end-start)/5)
            print(name)
            print('Accuracy: %.2f%%' % (np.mean(results)*100))
            print('Average Training Time:' + str((end-start)/5))

    fp = open('CNN_results.txt', mode='w')
    fp.write(str(log))

if __name__ == "__main__":
    svm()