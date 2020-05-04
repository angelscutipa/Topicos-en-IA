# Import pandas as pd
import pandas as pd
import numpy as np
import numpy as np 
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split

def  cal_cost(theta,X,y):

    m = len(y)
    
    hipotesis = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(hipotesis-y))
    return cost

def gradiente_des(X,y,theta,tasa_a,iteraciones):

    m = len(y)
    costo = np.zeros(iteraciones)
    for it in range(iteraciones):

        hipotesis = np.dot(X,theta)
        
        theta = theta -(1/m)*tasa_a*( X.T.dot((hipotesis - y)))
        costo[it]  = cal_cost(theta,X,y)
        
    return theta, costo

def fun(x, theta):
    return theta[1][0]*x + theta[0][0]

def leer_datos(nombre):
	precios = pd.read_csv(nombre)
	# Print out cars
	#print(precios)
	return precios

def aumentar_uno( l, n_filas_col ):
	lista=np.empty((n_filas_col[0], n_filas_col[1]+1))  # cuaido con esto es solo para esa cantidad de datos
	for i in range(0,n_filas_col[0]):
		lista[i][0]=i;
		for j in range(0,n_filas_col[1]):
			lista[i][j+1]=l[i][j];
	lista=lista.astype(float)
	return lista

def normalizacion(lista, n_filas_col):
	media_col=lista.mean(axis=0)  # MEDIA
	des_col=lista.std(axis=0)	# DESCVIACION ESTANDAR
	#print(media_col)		#Imprimir las medias por columna
	#print(des_col)			#imprimir la desviacion std

	l_normalizado=np.empty((n_filas_col[0], n_filas_col[1]))

	for i in range(0,n_filas_col[0]):
		for j in range(0,n_filas_col[1]):
			l_normalizado[i][j]= (lista[i][j+1]-media_col[j+1])/(des_col[j+1])

	#print("Datos normalizados")
	#print(l_normalizado)
	return l_normalizado

def div_train_prueba(l, l_normalizado):
	entrena=int(70*len(l)/100)
	entrenamiento = l_normalizado[0:entrena]
	prueba = l_normalizado[entrena:len(l)]
	#print("entrenamiento: ", entrenamiento)
	return entrenamiento, prueba

def div_X_Y(entrenamiento, n_filas_col):
	X=np.empty((len(entrenamiento),n_filas_col[1]))
	Y=np.empty((len(entrenamiento),1))

	for i in range(0,len(entrenamiento)):  # AL PARECER SE PUEDE AGREGAR DE OTRA MANERA
		X[i][0]=1
		for j in range(0,n_filas_col[1]-1):
			X[i][j+1]=entrenamiento[i][j]
		Y[i][0]=entrenamiento[i][n_filas_col[1]-1]
	#print("Datos divididos")
	#print(X)
	#print(Y)
	return X, Y

def ecuacion_normal(X, Y):
	theta_n=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)
	error=cal_cost(theta_n, X,Y)
	return theta_n, error

def aplicar_gradiente(X, Y, n_filas_col, alpha, iteraciones, theta_r):

	theta_r, costo = gradiente_des(X,Y,theta_r, alpha, iteraciones)
	return theta_r, costo

def ver_cuadro(Xtrain, Ytrain, theta_train, colors, metodo):
	ran=range(-2,4)
	plt.title("Grafica de "+metodo) 
	plt.xlabel("eje x") 
	plt.ylabel("eje y") 
	plt.plot(Xtrain[:, 1],Ytrain[:,0],"ob") 
	plt.plot(ran, [fun(i, theta_train) for i in ran], color=colors)
	plt.show()

def primer_ex(docs):

	for idx in range(0,len(docs)):
		datos_leido=leer_datos(docs[idx])
		n_filas_col=datos_leido.shape
		l=datos_leido.to_numpy()
		lista=aumentar_uno(l, n_filas_col)
		#NOMALIZACIÓN
		l_normalizado=normalizacion(lista, n_filas_col)
		_X, _Y=div_X_Y(l_normalizado, n_filas_col)
		#DIVIDIMOS EN CONJUNTO ENTRENAMIENTO Y PRUEBA
		# entrenamiento, prueba=div_train_prueba(l, l_normalizado)
		#DIVIDIMOS EN 2 PARTES
		Xtrain, Xtest, Ytrain, Ytest = train_test_split(_X, _Y, test_size=0.3, random_state=42)
		#Xtrain, Ytrain=div_X_Y(entrenamiento, n_filas_col)
		#Xtest, Ytest=div_X_Y(prueba, n_filas_col)
		# ECUANCION NORMAL
		"""xtrain = pd.read_csv('x_trainHome.csv',header=None)
		ytrain = pd.read_csv('y_trainHome.csv',header=None)
		X=xtrain.to_numpy()
		Y=ytrain.to_numpy()"""
		print("DATA SET: ", docs[idx])
		theta_train_n, error_train=ecuacion_normal(Xtrain,Ytrain)
		print("ECUACION NORMAL CON DATOS DE ENTRENAMIENTO")
		print("Respuesta: ")
		print( theta_train_n )
		print("Error: ")
		print( error_train )

		theta_test_n, error_test=ecuacion_normal(Xtest,Ytest)
		print("ECUACION NORMAL EN DATOS DE PRUEBA")
		print("Respuesta: ")
		print( theta_test_n )
		print("Error: ")
		print( error_test )
		print("----------------------------------------------------------------------")

def segundo_ex(docs):
	for idx in range(0,len(docs)):
		datos_leido=leer_datos(docs[idx])
		n_filas_col=datos_leido.shape
		l=datos_leido.to_numpy()
		lista=aumentar_uno(l, n_filas_col)
		#NOMALIZACIÓN
		l_normalizado=normalizacion(lista, n_filas_col)
		_X, _Y=div_X_Y(l_normalizado, n_filas_col)
		#DIVIDIMOS EN CONJUNTO ENTRENAMIENTO Y PRUEBA
		# entrenamiento, prueba=div_train_prueba(l, l_normalizado)
		#DIVIDIMOS EN 2 PARTES
		Xtrain, Xtest, Ytrain, Ytest = train_test_split(_X, _Y, test_size=0.3, random_state=42)
		#Xtrain, Ytrain=div_X_Y(entrenamiento, n_filas_col)
		#Xtest, Ytest=div_X_Y(prueba, n_filas_col)
		# ECUANCION NORMAL
		"""xtrain = pd.read_csv('x_trainHome.csv',header=None)
		ytrain = pd.read_csv('y_trainHome.csv',header=None)
		X=xtrain.to_numpy()
		Y=ytrain.to_numpy()"""

		print("GRADIENTE DESCENDIENTE: ", docs[idx])

		alpha=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4]

		for j in range (0,len(alpha)):
			i=500
			while(i<3501):
				theta_r = np.random.randn(n_filas_col[1],1)
				for i in range(0,n_filas_col[1]):
					theta_r[i][0]=1
				theta_train_r, costo= aplicar_gradiente(Xtrain, Ytrain, n_filas_col, alpha[j], i, theta_r)
				#print("Respuesta: ")
				#print(theta_train_r )
				print("Costo final: ", i, " ", alpha[j])
				print(costo[len(costo)-1])
				i=i+500			    
			 
def tercer_exp():
	datos_leido=leer_datos("ex1data2(Home_1f).csv")
	n_filas_col=datos_leido.shape
	l=datos_leido.to_numpy()
	lista=aumentar_uno(l, n_filas_col)
	#NOMALIZACIÓN
	l_normalizado=normalizacion(lista, n_filas_col)
	_X, _Y=div_X_Y(l_normalizado, n_filas_col)
	#DIVIDIMOS EN CONJUNTO ENTRENAMIENTO Y PRUEBA
	# entrenamiento, prueba=div_train_prueba(l, l_normalizado)
	#DIVIDIMOS EN 2 PARTES
	Xtrain, Xtest, Ytrain, Ytest = train_test_split(_X, _Y, test_size=0.3, random_state=42)


	theta_train_n, error_train=ecuacion_normal(Xtrain,Ytrain)
	print("ECUACION NORMAL CON DATOS DE ENTRENAMIENTO")
	print("Respuesta: ")
	print( theta_train_n )
	print("Error: ")
	print( error_train )

	theta_test_n, error_test=ecuacion_normal(Xtest,Ytest)
	print("ECUACION NORMAL EN DATOS DE PRUEBA")
	print("Respuesta: ")
	print( theta_test_n )
	print("Error: ")
	print( error_test )

	#REGRESION LINEAL

	print("GRADIENTE DESCENDIENTE")

	alpha=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4]

	theta_r = np.random.randn(n_filas_col[1],1)
	for i in range(0,n_filas_col[1]):
		theta_r[i][0]=1

	theta_train_r, costo= aplicar_gradiente(Xtrain, Ytrain, n_filas_col, alpha[0], 1000)
	print("Respuesta: ")
	print(theta_train_r )
	print("Costo final: ")
	print(costo[len(costo)-1])


	ver_cuadro(Xtrain, Ytrain, theta_train_n, 'red','ecuacion normal')
	ver_cuadro(Xtrain, Ytrain, theta_train_r, 'yellow', 'gradiente descendiente')

def cuarto_exp(docs):
	for idx in range(0,len(docs)):
		datos_leido=leer_datos(docs[idx])
		n_filas_col=datos_leido.shape
		l=datos_leido.to_numpy()
		lista=aumentar_uno(l, n_filas_col)
		#NOMALIZACIÓN
		l_normalizado=normalizacion(lista, n_filas_col)
		_X, _Y=div_X_Y(l_normalizado, n_filas_col)
		#DIVIDIMOS EN CONJUNTO ENTRENAMIENTO Y PRUEBA
		# entrenamiento, prueba=div_train_prueba(l, l_normalizado)
		#DIVIDIMOS EN 2 PARTES
		Xtrain, Xtest, Ytrain, Ytest = train_test_split(_X, _Y, test_size=0.3, random_state=42)


		#REGRESION LINEAL

		print("GRADIENTE DESCENDIENTE")

		alpha=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4]

		theta_r = np.random.randn(n_filas_col[1],1)
		for i in range(0,n_filas_col[1]):
			theta_r[i][0]=1

		theta_train_r, costo_n= aplicar_gradiente(Xtrain, Ytrain, n_filas_col, alpha[0], 1500, theta_r)
		theta_train_r2, costo_p= aplicar_gradiente(Xtest, Ytest, n_filas_col, alpha[0], 1500, theta_train_r)

		x = np.arange(0,1500) 
		y = np.squeeze(np.asarray(costo_n))
		z = np.squeeze(np.asarray(costo_p))
		plt.title("Matplotlib demo") 
		plt.xlabel("x iteraciones") 
		plt.ylabel("y costo") 
		plt.plot(x,y) 
		plt.plot(x,z) 
		plt.show()


docs=["ex1data2(Home_1f).csv","oceano_simple.csv","petrol_consumption.csv"]

cuarto_exp(docs)

"""
for idx in range(0,len(docs)):
	datos_leido=leer_datos(docs[idx])
	n_filas_col=datos_leido.shape
	l=datos_leido.to_numpy()
	lista=aumentar_uno(l, n_filas_col)
	#NOMALIZACIÓN
	l_normalizado=normalizacion(lista, n_filas_col)
	_X, _Y=div_X_Y(l_normalizado, n_filas_col)
	#DIVIDIMOS EN CONJUNTO ENTRENAMIENTO Y PRUEBA
	# entrenamiento, prueba=div_train_prueba(l, l_normalizado)
	#DIVIDIMOS EN 2 PARTES
	Xtrain, Xtest, Ytrain, Ytest = train_test_split(_X, _Y, test_size=0.3, random_state=42)


	theta_train_n, error_train=ecuacion_normal(Xtrain,Ytrain)
	print("ECUACION NORMAL CON DATOS DE ENTRENAMIENTO")
	print("Respuesta: ")
	print( theta_train_n )
	print("Error: ")
	print( error_train )

	theta_test_n, error_test=ecuacion_normal(Xtest,Ytest)
	print("ECUACION NORMAL EN DATOS DE PRUEBA")
	print("Respuesta: ")
	print( theta_test_n )
	print("Error: ")
	print( error_test )

	#REGRESION LINEAL

	print("GRADIENTE DESCENDIENTE")

	alpha=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4]

	for j in range (0,len(alpha)):
	  i=500
	  while(i<3501):
	    theta_train_r, costo= aplicar_gradiente(Xtrain, Ytrain, n_filas_col, alpha, i)
		print("Respuesta: ")
		print(theta_train_r )
		print("Costo final: ")
		print(costo[len(costo)-1])
	    i=i+500


	ver_cuadro(Xtrain, Ytrain, theta_train_r, theta_train_n)"""