import numpy as np
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

data_url = 'https://personal.utdallas.edu/~sxs190355/occupancy_data.txt'


class RNN:
    def __init__(self, x, y, hidden_layers):
        self.hidden_layers = hidden_layers
        self.x = x
        self.y = y

        # Randomly assign initial weights
        self.u = np.random.randn(self.hidden_layers, self.x.shape[2])
        self.v = np.random.randn(self.hidden_layers, self.hidden_layers)
        self.w = np.random.randn(self.y.shape[1], self.hidden_layers)

    def forward_pass(self, instance):
        # Initialize the first hidden state vector
        hidden_states = np.zeros((self.hidden_layers, 1))
        self.hidden_states_list = [hidden_states]

        instance_x, instance_y = self.x[instance], self.y[instance]

        # Make use of hidden states
        self.input_list = []

        for instance in range(len(instance_x)):
            hidden_states, prediction = self.memory_cell(
                instance_x[instance], hidden_states)
            self.input_list.append(instance_x[instance].reshape(1, 1))
            self.hidden_states_list.append(hidden_states)

        self.error = prediction - instance_y
        self.square_err = 0.5*self.error**2
        self.yt = prediction

    def memory_cell(self, value, hidden_context):
        hidden_state = np.tanh(
            np.dot(self.v, hidden_context) + np.dot(self.u, value.reshape(1, 1)))
        prediction = np.dot(self.w, hidden_state)
        return hidden_state, prediction

    def backpropagation(self):
        dy = self.error
        du = np.zeros(self.u.shape)
        dv = np.zeros(self.v.shape)

        dw = np.dot(dy, self.hidden_states_list[-1].T)
        num_inputs = len(self.input_list)

        dh = np.dot(dy, self.w).T

        for state in reversed(range(num_inputs)):
            intermediate = (1-self.hidden_states_list[state+1]**2) * dh
            du += np.dot(intermediate, self.input_list[state].T)
            dv += np.dot(intermediate, self.hidden_states_list[state].T)

            dh = np.dot(self.v, intermediate)

        dw = np.clip(dw, -1, 1)
        self.w -= self.learning_rate * dw
        du = np.clip(du, -1, 1)
        self.u -= self.learning_rate * du
        dv = np.clip(dv, -1, 1)
        self.v -= self.learning_rate * dv

    def train(self, num_epochs, learning_rate):
        self.loss_record = []
        self.learning_rate = learning_rate

        for iteration in tqdm(range(num_epochs)):
            for instance in range(self.x.shape[0]):
                self.forward_pass(instance)
                self.backpropagation()

            self.loss_record.append(np.squeeze(
                self.square_err / self.x.shape[0]))

            self.square_err = 0

    def test(self, x, y):
        self.output_list = []
        self.predictions = []
        self.x = x
        self.y = y

        for instance in range(len(x)):
            self.forward_pass(instance)
            self.output_list.append(self.yt)

            if self.yt > 0.5:
                self.predictions.append(1)
            else:
                self.predictions.append(0)

        #!pip install scikit-plot
        #import scikitplot as skplt
        #skplt.metrics.plot_confusion_matrix(self.y, self.predictions)
        #skplt.metrics.plot_roc_curve(self.predictions, self.y)
        #skplt.metrics.plot_precision_recall_curve(self.y, self.predictions)

        conf_matrix = metrics.confusion_matrix(self.predictions, self.y)
        metrics.ConfusionMatrixDisplay(
            confusion_matrix=conf_matrix, display_labels=[False, True]).plot()
        # cm_display.plot()
        plt.show()

        print("Test Accuracy = ", metrics.accuracy_score(
            self.y, self.predictions))
        print("Precision = ", metrics.precision_score(self.y, self.predictions))
        print("Recall = ", metrics.recall_score(self.y, self.predictions))
        print("F1 Score = ", metrics.f1_score(self.y, self.predictions))


col_names = ['serial', 'date', 'temperature', 'humidity',
             'light', 'CO2', 'HumidityRatio', 'Occupancy']
# load dataset
raw_input = pd.read_csv(data_url,
                        header=None, names=col_names)
raw_input.drop(index=raw_input.index[0], axis=0, inplace=True)
raw_input.drop('serial', inplace=True, axis=1)
date_time = pd.to_datetime(raw_input.pop('date'), format='%Y.%m.%d %H:%M:%S')

mm = MinMaxScaler()
raw_input.fillna(0)
processed_data = pd.DataFrame(mm.fit_transform(
    raw_input), columns=raw_input.columns)

# Learning curve (accuracy vs sample size), roc, precision recall curve , accuracy vs epochs


correlation_matrix = processed_data.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()


ncols = len(processed_data.columns)
nrows = len(processed_data.index)


X = processed_data.iloc[:, 0:(ncols - 1)]

y = processed_data.iloc[:, (ncols-1)]

# X array dimensions [num_instances, timesteps, features]
X = np.array(X).reshape(len(y), 5, 1)
# Y array dimensions [num_instances, features]
y = np.array(y).reshape(len(y), 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y)

num_layers = 15
num_epochs = 100
learning_rate = 1e-2

rnn = RNN(X, y, num_layers)
rnn.train(num_epochs, learning_rate)
rnn.test(X_test, y_test)
