from tensorflow.keras import models
from tensorflow.keras import layers
from sklearn.svm import SVC

class SVM:
    def __init__(self, name, C, kernel, gamma):
        self.m = SVC(C=C, kernel=kernel, gamma=gamma, decision_function_shape='ovo', verbose=True, probability=True)
        self.name = name

    def train(self, train_x, train_y):
        self.m.fit(train_x, train_y)

    def evaluate(self, test_x, test_y):
        return self.m.score(test_x, test_y)

class CNN:
    def __init__(self, name, num_classes='26', in_shape=(28,28,1)):
        self.m = models.Sequential()
        self.name = name
        self.num_classes = num_classes
        self.in_shape = in_shape

    def build(self, f=32, k_size=(5,5), act='relu', p_size=(2,2)):
        self.m.add(layers.Conv2D(f, kernel_size=k_size, activation=act, input_shape=self.in_shape))
        self.m.add(layers.MaxPool2D(pool_size=p_size))
        self.m.add(layers.Flatten())
        self.m.add(layers.Dense(128, activation='relu'))
        self.m.add(layers.Dense(self.num_classes, activation='softmax'))

    def compile(self, opt='sgd', l='mse', m=['accuracy']):
        self.m.compile(optimizer=opt,
                loss=l,
                metrics=m)

    def train(self, train_x, train_y, b_size=1, ep=5):
        x = train_x.reshape(train_x.shape[0], *self.in_shape)
        y = train_y

        self.m.fit(x, y, batch_size=b_size, epochs=ep)
        self.m.save('models/' + self.name + '.h5')

    def __call__(self):
        return self.m

class MLP:
    def __init__(self, name, num_classes='26', in_shape=(28,28,1)):
        self.m = models.Sequential()
        self.name = name
        self.num_classes = num_classes
        self.in_shape = in_shape

    def build(self, act='relu', dropout_rate=0):
        self.m.add(layers.Dense(784, activation='relu'))
        self.m.add(layers.Dense(128, activation='relu'))
        self.m.add(layers.Dense(128, activation='relu'))
        self.m.add(layers.Dropout(dropout_rate))
        self.m.add(layers.Dense(128, activation='relu'))
        self.m.add(layers.Dropout(dropout_rate))
        self.m.add(layers.Dense(self.num_classes, activation='softmax'))

    def compile(self, opt='sgd', l='mse', m=['accuracy']):
        self.m.compile(optimizer=opt,
                loss=l,
                metrics=m)

    def train(self, train_x, train_y, b_size=1, ep=5):
        x = train_x
        y = train_y

        self.m.fit(x, y, batch_size=b_size, epochs=ep)
        self.m.save('models/' + self.name + '.h5')

    def __call__(self):
        return self.m

def evaluate(name, type, test_x, test_y):
    m = models.load_model('models/' + name + '.h5')
    if type == 'cnn':
        x = test_x.reshape(test_x.shape[0], 28,28,1)
    else:
        x = test_x
    y = test_y

    return m.evaluate(x, y)
