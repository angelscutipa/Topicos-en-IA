import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import sys

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def Leer_Datos(file_datos):
    return pd.read_csv(file_datos)

def Normalizar_Datos(data):  
    mean_data = np.mean(data)
    standard_dev = np.std(data)
    data = data - mean_data
    data = data / standard_dev
    return data, mean_data, standard_dev

def Sigmoidal(X, W):
    result = np.matmul(X, W)
    result = np.exp(-result)
    result = result + 1
    result = 1 / result
    return result

def Separar_X_y(data):
    n = data.shape[1]
    X = data[:, :n-1]
    y = data[:, n-1:]
    return X, y

def Crear_Entrenamiento_Prueba(data):
    num_rows = data.shape[0]
    porcentaje_entrenamiento = 0.7
    filas_a_dividir = int(num_rows * porcentaje_entrenamiento)
    entrenamiento, prueba = data[:filas_a_dividir, :], data[filas_a_dividir:, :]
    return entrenamiento, prueba

def Crear_Pesos(X):
    return np.random.rand(X.shape[1])

def Calcular_Funcion_Costo(X, y, W):
    pred = Sigmoidal(X, W)
    result = np.sum( (y * np.log(pred)) + ((1-y) * np.log(1-pred)) )
    result = (-result)/X.shape[0]
    return result

def Calcular_Gradiente(X, y, pred):
    gradient = pred - y
    gradient = np.matmul(np.transpose(X), gradient)
    gradient = np.divide(gradient, y.shape[0])
    return gradient

def Gradiente_Descendiente(X, y, W, num_iter, val_alpha):
    val_costos = np.zeros(num_iter)
    for i in range(num_iter):
        pred = Sigmoidal(X, W)
        gradient = Calcular_Gradiente(X, y, pred)
        result = np.multiply(gradient, val_alpha)
        W = W - result
        val_costos[i] = Calcular_Funcion_Costo(X, y, W)
    return W, val_costos

def Calcular_Accuraccy(X, y, W):
    pred = Sigmoidal(X, W)
    pred = pred > 0.5
    result = np.logical_xor(np.logical_not(pred), y)
    return np.sum(result) / y.shape[0]
    
def Crear_k_folds(data, k):
    np.random.shuffle(data)
    tam_fold = int(data.shape[0] / k)
    recordar_tam_fold = int(data.shape[0] % k)
    data = data[:data.shape[0]-recordar_tam_fold,:]
    k_folds = []
    idx_row = 0
    for i in range(k):
        X, y = Separar_X_y(data[idx_row:idx_row+tam_fold, :])
        k_folds.append({"X": X, "y" : y})
        idx_row += tam_fold
    return k_folds, tam_fold




data_files = ["cardiaca.csv", "diabetes.csv"]
num_iters = [500, 1000, 1500, 2000, 2500, 3000, 3500]
num_iters_label = num_iters.copy()
num_iters_label.insert(0, "T. APRENDIZAJE \ ITERACIONES")
alpha = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
k = 3

for datos in data_files:
    result_table = [alpha]
    data = Leer_Datos(datos)
    X, y = Separar_X_y(data.values)
    norm_data_X, prom_data_X, standard_dev_X = Normalizar_Datos(X)
    norm_data = np.concatenate((norm_data_X, y), axis=1)
    k_folds, tam_fold = Crear_k_folds(norm_data, k)
    for num_iter in num_iters:
        val_alpha_fila = []
        for val_alpha in alpha:
            acc_test_total = 0.0
            for i in range(k):
                X_train = np.zeros((tam_fold * (k-1), norm_data.shape[1] - 1))
                X_test = np.zeros((tam_fold, norm_data.shape[1] - 1))
                y_train = np.zeros((tam_fold * (k-1), 1))
                y_test = np.zeros((tam_fold, 1))
                cont_sz_fold = 0
                for j in range(k):
                    if j == i:
                        X_test = k_folds[i]['X']
                        y_test = k_folds[i]['y']
                    else:
                        X_train[cont_sz_fold:cont_sz_fold+tam_fold, :] = k_folds[j]['X']
                        y_train[cont_sz_fold:cont_sz_fold+tam_fold, :] = k_folds[j]['y']
                        cont_sz_fold += tam_fold

                y_train = np.reshape(y_train, y_train.shape[0])
                y_test = np.reshape(y_test, y_test.shape[0])

                X_train = np.c_[X_train, np.ones(X_train.shape[0])]     #bias
                X_test = np.c_[X_test, np.ones(X_test.shape[0])]        #bias
                W = Crear_Pesos(X_train)
                W, val_costos = Gradiente_Descendiente(X_train, y_train, W, num_iter, val_alpha)
                acc_test = Calcular_Accuraccy(X_test, y_test, W)
                acc_test_total += acc_test

                #print(cost_test)
                #print(W)
            acc_test_total /= k
            val_alpha_fila.append("%.4f" % acc_test_total)
        result_table.append(val_alpha_fila)

    headerColor = 'grey'
    rowEvenColor = 'lightgrey'
    rowOddColor = 'white'

    fig = go.Figure(data=[go.Table(
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
        ))
    ])
    fig.update_layout(
    title=go.layout.Title(
        text=datos,
        xref="paper",
        x=0
    ))
    fig.show()




data_files = ["cardiaca.csv", "diabetes.csv"]
num_iters_label = num_iters.copy()
num_iters_label.insert(0, "Tasas de aprendizaje \ Numero de iteraciones")
k = 3


for datos in data_files:
    result_table = [alpha]
    data = Leer_Datos(datos)
    X, y = Separar_X_y(data.values)
    norm_data_X, prom_data_X, standard_dev_X = Normalizar_Datos(X)
    norm_data = np.concatenate((norm_data_X, y), axis=1)
    k_folds, tam_fold = Crear_k_folds(norm_data, k)

    val_alpha_fila = []

    plotcos=np.empty((3,2000))
    acc_test_total = 0.0
    for i in range(k):
        X_train = np.zeros((tam_fold * (k-1), norm_data.shape[1] - 1))
        X_test = np.zeros((tam_fold, norm_data.shape[1] - 1))
        y_train = np.zeros((tam_fold * (k-1), 1))
        y_test = np.zeros((tam_fold, 1))
        cont_sz_fold = 0
        for j in range(k):
            if j == i:
                X_test = k_folds[i]['X']
                y_test = k_folds[i]['y']
            else:
                X_train[cont_sz_fold:cont_sz_fold+tam_fold, :] = k_folds[j]['X']
                y_train[cont_sz_fold:cont_sz_fold+tam_fold, :] = k_folds[j]['y']
                cont_sz_fold += tam_fold

        y_train = np.reshape(y_train, y_train.shape[0])
        y_test = np.reshape(y_test, y_test.shape[0])

        X_train = np.c_[X_train, np.ones(X_train.shape[0])]     #bias
        X_test = np.c_[X_test, np.ones(X_test.shape[0])]        #bias
        W = Crear_Pesos(X_train)
        W, val_costos = Gradiente_Descendiente(X_train, y_train, W, 2000, 0.01)
        plotcos[i]=val_costos;
        """plt.title(datos)
        plt.plot(range(len(val_costos)), val_costos)
        plt.show()"""
        
    plt.title(datos)
    plt.plot(range(len(plotcos[0])), plotcos[0])
    plt.plot(range(len(plotcos[1])), plotcos[1])
    plt.plot(range(len(plotcos[2])), plotcos[2])
    plt.show()