import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#prep model
architecture = [
    {"input_dim": 2, "output_dim": 13, "activation": "relu"},
    {"input_dim": 13, "output_dim": 1, "activation": "sigmoid"},
]
#currently .853 acc model


def initLayers(architecture, seed=99):  # seed makes behavior of initializer deterministic
    np.random.seed(seed)
    numLayers = len(architecture)
    params = {}
    for idx, layer in enumerate(architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        params['W' + str(layer_idx)] = np.random.randn(layer_output_size, layer_input_size) * 0.1
        params['b' + str(layer_idx)] = np.random.randn(layer_output_size, 1) * 0.1
        
    return params

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_back(dA, Z):
    sig = sigmoid(Z)
    return (dA * sig * (1-sig))

def relu_back(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z<=0] = 0;
    return dZ

def convert_prob_into_class(probs):
    return (probs > 0.5).astype(int)

def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    if activation == "relu":
        activation_func = relu
    elif activation == "sigmoid":
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activ function')
        
    return activation_func(Z_curr), Z_curr

def full_forward_propagation(X, params_values, nn_architecture):
    memory = {}
    A_curr = X
    
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr
        activ_function_curr = layer["activation"]
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr
       
    return A_curr, memory

def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1]
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()


def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    m = A_prev.shape[1]
    if activation == "relu":
        backward_activation_func = relu_back
    elif activation == "sigmoid":
        backward_activation_func = sigmoid_back
    else:
        raise Exception('Non-supported activation function')
    
    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr


def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
    
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer["activation"]
        dA_curr = dA_prev
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
    
    return grads_values

def update(params_values, grads_values, nn_architecture, learning_rate):
    for layer_idx in range(len(nn_architecture)):
        params_values["W" + str(layer_idx + 1)] -= learning_rate * grads_values["dW" + str(layer_idx + 1)]
        params_values["b" + str(layer_idx + 1)] -= learning_rate * grads_values["db" + str(layer_idx + 1)]
    return params_values;


def train(X, Y, nn_architecture, epochs, learning_rate):
    params_values = initLayers(nn_architecture, 2)
    cost_history = []
    accuracy_history = []
    
    for i in range(epochs):
        Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)
        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)
        
    return params_values, cost_history, accuracy_history

#model data
def generate_data():
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=100)
    y = y.reshape(y.shape[0], 1)
    return X, y

def prepare_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train.T, X_test.T, y_train.T, y_test.T

def plot_results(cost_history, accuracy_history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(cost_history, label='Cost')
    plt.title('Cost over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history, label='Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def evaluate_model(X_test, y_test, params_values, nn_architecture):
    Y_test_hat, _ = full_forward_propagation(X_test, params_values, nn_architecture)
    test_cost = get_cost_value(Y_test_hat, y_test)
    test_accuracy = get_accuracy_value(Y_test_hat, y_test)
    return test_cost, test_accuracy

def main():
    try:
        print("Close the Graphs for Test Cost/Test Acc")
        X, y = generate_data()
        X_train, X_test, y_train, y_test = prepare_data(X, y)
        print(X_train.shape)  # this needs to be (2, 700), debugging
        params_values, cost_history, accuracy_history = train(X_train, y_train, architecture, epochs=1000, learning_rate=0.01)
        plot_results(cost_history, accuracy_history)

        test_cost, test_accuracy = evaluate_model(X_test, y_test, params_values, architecture)
        print(f"Test Cost: {test_cost}")
        print(f"Test Accuracy: {test_accuracy}")
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()


