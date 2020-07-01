# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import sys

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def Leer_Datos(file_name):
    return pd.read_csv(file_name)

def Normalizar_Datos(data):
    mean_data = np.mean(data)
    standard_dev = np.std(data)
    data = data - mean_data
    data = data / standard_dev
    return data, mean_data, standard_dev

def Sigmoidal(Z):
    A = np.exp(-Z)
    A = 1 + A
    A = 1 / A
    return A

def Activacion(X, W, b):
    Z = np.matmul(W, X) + b
    A = Sigmoidal(Z)
    return A, Z

def Calcular_Accuracy(X, y, W_dict, b_dict):
    _, A_dict = Forward(X, W_dict, b_dict)
    A_L = A_dict["A%d"%(len(A_dict)-1)]
    result = A_L > 0.5
    result = np.logical_xor(np.logical_not(result), y)
    return np.sum(result) / y.shape[1]

def Separar_X_y(data):
    n = data.shape[0]
    X = data[:n-1, :]
    y = data[n-1:, :]
    return X, y

def Crear_k_folds(data, k):
    size_fold = int(data.shape[1] / k)
    remainder_size_fold = int(data.shape[1] % k)
    data = data[:,:data.shape[1]-remainder_size_fold]
    k_folds = []
    idx_col = 0
    for i in range(k):
        X, y = Separar_X_y(data[:, idx_col:idx_col+size_fold])
        k_folds.append({"X": X, "y" : y})
        idx_col += size_fold
    return k_folds, size_fold

def Crear_Entrenamiento_Prueba(data):
    num_cols = data.shape[1]
    train_percentage = 0.7
    col_split_data = int(num_cols * train_percentage)
    training, test = data[:, :col_split_data], data[:, col_split_data:]
    return training, test

def Crear_Bias(n_actual):
    return np.random.rand(n_actual, 1)

def Crear_Pesos(n_anterior, n_actual):
    return np.random.rand(n_actual, n_anterior)

def Crear_W_b_dict(model_mlp):
    W_dict = {}
    b_dict = {}
    for l in range(len(model_mlp)-1):
        W_dict["W%d"%(l+1)] = Crear_Pesos(model_mlp[l], model_mlp[l+1])
        b_dict["b%d"%(l+1)] = Crear_Bias(model_mlp[l+1])
    return W_dict, b_dict

def Calcular_Funcion_Costo(A_L, y):
    result = np.sum( (y * np.log(A_L)) + ((1-y) * np.log(1-A_L)) )
    result = (-result)/y.shape[1]
    return result

def dS(Z):
    s = Sigmoidal(Z)
    return np.multiply(s, 1 - s)

def Forward(X, W_dict, b_dict):
    A = X
    Z_dict = {}
    A_dict = {"A0":A}
    for l in range(len(W_dict)):
        A, Z = Activacion( A, W_dict[ "W%d"%(l+1) ], b_dict[ "b%d"%(l+1) ] )
        Z_dict["Z%d"%(l+1)] = Z
        A_dict["A%d"%(l+1)] = A
    return Z_dict, A_dict

def get_dC_dA_L(A_L, y):
    return -np.divide(y, A_L) + np.divide(1-y,1-A_L)

def Backward(W_dict, b_dict, Z_dict, A_dict, y, learn_rate):
    m = A_dict["A0"].shape[1]
    dA = get_dC_dA_L(A_dict["A%d"%(len(A_dict)-1)], y)
    for l in reversed(range(len(W_dict))):
        dZ = np.multiply(dA, dS(Z_dict["Z%d"%(l+1)]))
        dW = np.divide(np.dot( dZ, A_dict["A%d"%(l)].T ), m)
        db = np.divide(np.sum( dZ, axis=1, keepdims=True ), m)
        dA = np.dot( W_dict["W%d"%(l+1)].T, dZ )
        W_dict["W%d"%(l+1)] = W_dict["W%d"%(l+1)] - np.multiply(learn_rate, dW)
        b_dict["b%d"%(l+1)] = b_dict["b%d"%(l+1)] - np.multiply(learn_rate, db)
    return W_dict, b_dict

def Gradiente_Descendiente(X, y, W_dict, b_dict, num_iter, learn_rate):
    costs = np.zeros(num_iter)
    for i in range(num_iter):
        Z_dict, A_dict = Forward(X, W_dict, b_dict)
        W_dict, b_dict = Backward(W_dict, b_dict, Z_dict, A_dict, y, learn_rate)
        costs[i] = Calcular_Funcion_Costo(A_dict["A%d"%(len(A_dict)-1)], y)
    return W_dict, b_dict, costs

"""# Experimento 1 - heart.csv"""

data_files = ["heart.csv"]
learn_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
num_iters = [500, 1000, 1500, 2000, 2500, 3000, 3500]
list_hidden_layers = [[], [2], [3, 2], [4, 3, 2]]

num_iters_label = num_iters.copy()
num_iters_label.insert(0, "Alpha \ N° Ite")

k = 3

for name in data_files:
    for hidden_layers in list_hidden_layers:        
        result_table = [learn_rates]
        data = Leer_Datos(name)
        data_np = data.values
        np.random.shuffle(data_np)
        transposed_data = np.transpose(data_np)
        X, y = Separar_X_y(transposed_data)
        norm_data_X, mean_data_X, standard_dev_X = Normalizar_Datos(X)
        norm_data = np.concatenate((norm_data_X, y), axis=0)
        k_folds, size_fold = Crear_k_folds(norm_data, k)
        for num_iter in num_iters:
            learn_rate_row = []
            for learn_rate in learn_rates:
                acc_test_total = 0.0
                for i in range(k):
                    X_train = np.zeros((norm_data.shape[0] - 1, size_fold * (k-1)))
                    X_test = np.zeros((norm_data.shape[0] - 1, size_fold))
                    y_train = np.zeros((1, size_fold * (k-1)))
                    y_test = np.zeros((1, size_fold))
                    count_sz_fold = 0
                    for j in range(k):
                        if j == i:
                            X_test = k_folds[i]['X']
                            y_test = k_folds[i]['y']
                        else:
                            X_train[:, count_sz_fold:count_sz_fold+size_fold] = k_folds[j]['X']
                            y_train[:, count_sz_fold:count_sz_fold+size_fold] = k_folds[j]['y']
                            count_sz_fold += size_fold
                    
                    model_mlp = hidden_layers.copy()
                    model_mlp.insert(0, X_train.shape[0])
                    model_mlp.append(1)
                    
                    W_dict, b_dict = Crear_W_b_dict(model_mlp)
                    W_dict, b_dict, costs = Gradiente_Descendiente(X_train, y_train, W_dict, b_dict, num_iter, learn_rate)
                    acc_test = Calcular_Accuracy(X_test, y_test, W_dict, b_dict)
                    acc_test_total += acc_test

                acc_test_total /= k
                learn_rate_row.append("%.4f" % acc_test_total)
            result_table.append(learn_rate_row)

        headerColor = 'grey'
        rowEvenColor = 'lightgrey'
        rowOddColor = 'white'

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=num_iters_label,
                        line_color='darkslategray',
                        fill_color=headerColor,
                        align=['left','center'],
                        font=dict(color='white', size=12)
                    ),
                    cells=dict(
                        values=result_table,
                        line_color='darkslategray',
                        fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor,rowEvenColor]*6],
                        align = ['left', 'center'],
                        font = dict(color = 'darkslategray', size = 11)
                    )
                )
            ]
        )
        fig.update_layout(
            title=go.layout.Title(
                text=name + "; Capas Ocultas = " + str(model_mlp),
                xref="paper",
                x=0
            )
        )
        fig.show()


"""# Experimento 1 - Iris.csv"""

data_files = ["Iris.csv"]
learn_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
num_iters = [500, 1000, 1500, 2000, 2500, 3000, 3500]
list_hidden_layers = [[], [2], [3, 2], [4, 3, 2]]

num_iters_label = num_iters.copy()
num_iters_label.insert(0, "Alpha \ N° Ite")

k = 3

for name in data_files:
    for hidden_layers in list_hidden_layers:        
        result_table = [learn_rates]
        data = Leer_Datos(name)
        data_np = data.values
        print(data_np)
        np.random.shuffle(data_np)
        transposed_data = np.transpose(data_np)
        X, y = Separar_X_y(transposed_data)
        norm_data_X, mean_data_X, standard_dev_X = Normalizar_Datos(X)
        norm_data = np.concatenate((norm_data_X, y), axis=0)
        k_folds, size_fold = Crear_k_folds(norm_data, k)
        for num_iter in num_iters:
            learn_rate_row = []
            for learn_rate in learn_rates:
                acc_test_total = 0.0
                for i in range(k):
                    X_train = np.zeros((norm_data.shape[0] - 1, size_fold * (k-1)))
                    X_test = np.zeros((norm_data.shape[0] - 1, size_fold))
                    y_train = np.zeros((1, size_fold * (k-1)))
                    y_test = np.zeros((1, size_fold))
                    count_sz_fold = 0
                    for j in range(k):
                        if j == i:
                            X_test = k_folds[i]['X']
                            y_test = k_folds[i]['y']
                        else:
                            X_train[:, count_sz_fold:count_sz_fold+size_fold] = k_folds[j]['X']
                            y_train[:, count_sz_fold:count_sz_fold+size_fold] = k_folds[j]['y']
                            count_sz_fold += size_fold
                    
                    model_mlp = hidden_layers.copy()
                    model_mlp.insert(0, X_train.shape[0])
                    model_mlp.append(1)
                    
                    W_dict, b_dict = Crear_W_b_dict(model_mlp)
                    W_dict, b_dict, costs = Gradiente_Descendiente(X_train, y_train, W_dict, b_dict, num_iter, learn_rate)
                    acc_test = Calcular_Accuracy(X_test, y_test, W_dict, b_dict)
                    acc_test_total += acc_test

                acc_test_total /= k
                learn_rate_row.append("%.4f" % acc_test_total)
            result_table.append(learn_rate_row)

        headerColor = 'grey'
        rowEvenColor = 'lightgrey'
        rowOddColor = 'white'

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=num_iters_label,
                        line_color='darkslategray',
                        fill_color=headerColor,
                        align=['left','center'],
                        font=dict(color='white', size=12)
                    ),
                    cells=dict(
                        values=result_table,
                        line_color='darkslategray',
                        fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor,rowEvenColor]*6],
                        align = ['left', 'center'],
                        font = dict(color = 'darkslategray', size = 11)
                    )
                )
            ]
        )
        fig.update_layout(
            title=go.layout.Title(
                text=name + "; Capas Ocultas = " + str(model_mlp),
                xref="paper",
                x=0
            )
        )
        fig.show()

from google.colab import drive
drive.mount('/content/drive')

"""

# Experimento 2 - heart.csv
"""

from sklearn import svm

def Separar_X_y1(data):
    n = data.shape[1]
    X = data[:, :n-1]
    y = data[:, n-1:]
    return X, y

def Crear_k_folds1(data, k):
    np.random.shuffle(data)
    size_fold = int(data.shape[0] / k)
    remainder_size_fold = int(data.shape[0] % k)
    data = data[:data.shape[0]-remainder_size_fold,:]
    k_folds = []
    idx_row = 0
    for i in range(k):
        X, y = Separar_X_y1(data[idx_row:idx_row+size_fold, :])
        k_folds.append({"X": X, "y" : y})
        idx_row += size_fold
    return k_folds, size_fold


kernels = ['linear', 'poly', 'rbf']
C = [2500.0, 1000.0, 100.0, 1.0, 0.5, 0.01]

C_label = C.copy()
C_label.insert(0, "Kernels \ C")
k = 3
       
result_table = [kernels]
data_file = "heart.csv"
data = Leer_Datos(data_file)
data_np = data.values
np.random.shuffle(data_np)
X, y = Separar_X_y1(data_np)
norm_data_X, mean_data_X, standard_dev_X = Normalizar_Datos(X)
norm_data = np.concatenate((norm_data_X, y), axis=1)
k_folds, size_fold = Crear_k_folds1(norm_data, k)
for c in C:
    kernel_row = []
    for kernel in kernels:
        acc_test_total = 0.0
        for i in range(k):
            X_train = np.zeros((size_fold * (k-1), norm_data.shape[1] - 1))
            X_test = np.zeros((size_fold, norm_data.shape[1] - 1))
            y_train = np.zeros((size_fold * (k-1), 1))
            y_test = np.zeros((size_fold, 1))
            count_sz_fold = 0
            for j in range(k):
                if j == i:
                    X_test = k_folds[i]['X']
                    y_test = k_folds[i]['y']
                else:
                    X_train[count_sz_fold:count_sz_fold+size_fold, :] = k_folds[j]['X']
                    y_train[count_sz_fold:count_sz_fold+size_fold, :] = k_folds[j]['y']
                    count_sz_fold += size_fold

            y_train = np.reshape(y_train, y_train.shape[0])
            y_test = np.reshape(y_test, y_test.shape[0])

            clf = svm.SVC(C=c, kernel=kernel, gamma='scale')
            clf.fit(X_train, y_train)              
            acc_test = clf.score(X_test, y_test)
            acc_test_total += acc_test

        acc_test_total /= k
        kernel_row.append("%.4f" % acc_test_total)
    result_table.append(kernel_row)

headerColor = 'grey'
rowEvenColor = 'lightgrey'
rowOddColor = 'white'

fig = go.Figure(
    data=[
        go.Table(
            header=dict(
                values=C_label,
                line_color='darkslategray',
                fill_color=headerColor,
                align=['left','center'],
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=result_table,
                line_color='darkslategray',
                fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor,rowEvenColor]*6],
                align = ['left', 'center'],
                font = dict(color = 'darkslategray', size = 11)
            )
        )
    ]
)
fig.update_layout(
    title=go.layout.Title(
        text=data_file,
        xref="paper",
        x=0
    )
)
fig.show()

"""# Experimento 2 - Iris.csv"""

from sklearn import svm

def Separar_X_y1(data):
    n = data.shape[1]
    X = data[:, :n-1]
    y = data[:, n-1:]
    return X, y

def Crear_k_folds1(data, k):
    np.random.shuffle(data)
    size_fold = int(data.shape[0] / k)
    remainder_size_fold = int(data.shape[0] % k)
    data = data[:data.shape[0]-remainder_size_fold,:]
    k_folds = []
    idx_row = 0
    for i in range(k):
        X, y = Separar_X_y1(data[idx_row:idx_row+size_fold, :])
        k_folds.append({"X": X, "y" : y})
        idx_row += size_fold
    return k_folds, size_fold

#np.random.seed(0)

kernels = ['linear', 'poly', 'rbf']
C = [2500.0, 1000.0, 100.0, 1.0, 0.5, 0.01]

C_label = C.copy()
C_label.insert(0, "Kernels \ C")
k = 3
       
result_table = [kernels]
data_file = "Iris2.csv"
data = Leer_Datos(data_file)
data_np = data.values
np.random.shuffle(data_np)
X, y = Separar_X_y1(data_np)
norm_data_X, mean_data_X, standard_dev_X = Normalizar_Datos(X)
norm_data = np.concatenate((norm_data_X, y), axis=1)
k_folds, size_fold = Crear_k_folds1(norm_data, k)
for c in C:
    kernel_row = []
    for kernel in kernels:
        acc_test_total = 0.0
        for i in range(k):
            X_train = np.zeros((size_fold * (k-1), norm_data.shape[1] - 1))
            X_test = np.zeros((size_fold, norm_data.shape[1] - 1))
            y_train = np.empty(shape=(size_fold * (k-1), 1), dtype=object)
            y_test = np.empty(shape=(size_fold, 1), dtype=object)
            count_sz_fold = 0
            for j in range(k):
                if j == i:
                    X_test = k_folds[i]['X']
                    y_test = k_folds[i]['y']
                else:
                    X_train[count_sz_fold:count_sz_fold+size_fold, :] = k_folds[j]['X']
                    y_train[count_sz_fold:count_sz_fold+size_fold, :] = k_folds[j]['y']
                    count_sz_fold += size_fold

            y_train = np.reshape(y_train, y_train.shape[0])
            y_test = np.reshape(y_test, y_test.shape[0])

            clf = svm.SVC(C=c, kernel=kernel, gamma='scale')
            clf.fit(X_train, y_train)              
            acc_test = clf.score(X_test, y_test)
            acc_test_total += acc_test

        acc_test_total /= k
        kernel_row.append("%.4f" % acc_test_total)
    result_table.append(kernel_row)

headerColor = 'grey'
rowEvenColor = 'lightgrey'
rowOddColor = 'white'

fig = go.Figure(
    data=[
        go.Table(
            header=dict(
                values=C_label,
                line_color='darkslategray',
                fill_color=headerColor,
                align=['left','center'],
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=result_table,
                line_color='darkslategray',
                fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor,rowEvenColor]*6],
                align = ['left', 'center'],
                font = dict(color = 'darkslategray', size = 11)
            )
        )
    ]
)
fig.update_layout(
    title=go.layout.Title(
        text=data_file,
        xref="paper",
        x=0
    )
)
fig.show()
