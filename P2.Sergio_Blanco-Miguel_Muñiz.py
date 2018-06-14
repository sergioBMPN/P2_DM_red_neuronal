import numpy as np
import os
#import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#import tensorflow as tf
np.random.seed()
 
class xorMLP(object):
    def __init__(self, learning_rate=0.):
        #En el constructor de la clase guardamos la constante aprendizaje que se nos ha proporcionado
        #Asi como también el valor de las entradas, que será el array de las X, con sus correspondientes salidas esperadas que serán las Y
        #Por otro lado asignamos las bias de las neuronas con valor 1 o -1, en este caso hemos escogido el valor -1
        self.aprendizaje=learning_rate
        self.X = np.array([[0,0],[0,1],[1,0],[1,1]])
        self.Y = np.array([[0],[1],[1],[0]])
        self.bias=-1

        # Asignamos unos valores aleatorios (entre -1 y 1) para los pesos de las entradas que van a la neurona de la capa oculta
        rand = np.random.uniform(-1, 1, size=9)
        rand = rand.round(5)
        #capaOculta
        self.hide_w = np.array([(rand[0], rand[1],rand[2]), (rand[3],rand[4],rand[5])])

        # Asignamos unos valores aleatorios (entre -1 y 1) para los pesos de las entradas que van a la neurona de la capa de salida
        self.output_w = np.array([(rand[6],rand[7],rand[8])])


    def fit(self):

        #Establecemos un valor para el error pequeño, que será la condición de parada de entrenamiento de la red
        #En el caso de que la suma (en valor absoluto) de los errores para los 4 casos es más pequeño que el error que hemos establecido
        #quiere decir que la red de neuronas esta entrenada con cierta precisión
        errorMAX=1e-2
        sumaEO = errorMAX
        k=0 #Variable que determinará cuantas iteracciones se han requerido para obtener el resultado
        done=False
       
        while not done:
              if sumaEO < errorMAX:#Condición de parada del entrenamiento de la red
                done=True

              else:
                    #Comienza el proceso de Feedfordward
                    sumaEO = 0

                    #Con este bucle se calcula por un lado  el sumatorio de todas las entradas de la neurona de la capa oculta por sus correspondientes pesos
                    #para calcular la función activación de esta neurona, que en este caso será la sigmoidal
                    for x,y in zip(self.X,self.Y):
                        x_temp=np.array(np.concatenate(([1],x)))
                        fAct_h = np.array([])
                        for neurona in self.hide_w:
                            temp = np.array([])
                            for peso in range(1, len(neurona), 1):
                                temp = np.append(fAct_h, 1 / (1 + np.exp(- np.dot(neurona, x_temp))))
                            fAct_h = np.append(fAct_h,sum(temp))

                        fAct_o = np.array([])
                        fAct_h_temp=np.array(np.concatenate(([1],fAct_h)))
                        for neurona in self.output_w:
                            temp = np.array([])
                            for peso in range(1, len(neurona), 1):
                                temp = np.append(fAct_o, 1 / (1 + np.exp(- np.dot(neurona, fAct_h_temp))))
                            fAct_o = np.append(fAct_o,sum(temp))


                        #Calculamos el error que hemos obtenido con el valor de salida de la neurona de salida, y la comparamos con el valor esperado y
                        eO=np.array([])
                        for neurona in fAct_o:
                            eO= np.append(eO,(neurona*(1-neurona))*(y[0] - neurona))
                        sumaEO += sum(abs(eO))#Actualizamos el valor de la suma de los errores de salida

                        #A continuación comienza el proceso de BackPropagation
                        old_output_w=self.output_w#Guardamos el valor del peso de la salida de la neurona de la capa de salida, ya que posteriormente lo tendremos que utilizar para calcular el error de la capa oculta

                        #incWho
                        #Actualizamos los pesos que van de las neuronas de la capa oculta a la neurona de la capa de salida
                        for neurona in range(0, len(self.output_w), 1):
                            for peso in range(1, len(self.output_w[neurona]), 1):
                                self.output_w[neurona][peso] += self.aprendizaje * eO * fAct_h[neurona]

                        ###############Es necesario incWio  porque no hay conexion directa i-o????????


                        #Calculamos el error de la capa oculta con el valor del peso que habiamos guardado previamente antes de actualizarlo
                        eh = np.array([])
                        for neurona in range(0,len(fAct_h),1):
                            eh = np.append(eh, (fAct_h[neurona] * (1 - fAct_h[neurona])) * (eO * old_output_w[0][neurona+1]))

                        old_hide_w=self.hide_w

                        #incWih
                        #Con el error de la capa oculta, a continuación, actualizamos los pesos que van de las neurona de la capa de entrada a la neurona de la capa oculta, así como tambien peso del bias
                        temp=np.array(np.concatenate(([0],x)))
                        for neurona in range(0, len(self.hide_w), 1):
                            for peso in range(0, len(self.hide_w[neurona]), 1):
                                self.hide_w[neurona][peso] += self.aprendizaje * eh[neurona] * x[neurona]


                        k+=1

        print ("Iteraciones totales: ",k)

 
    def predict(self, x):
        #Una vez que hemos terminado de entrenar a la red de neuronas, para comprobar la precisión de la red,
        #en el predict, le pasamos como parámetro el array con las entradas y así obtener una salida estimada
        """
        x = [x1, x2]
        """
        #Proceso para obtener el valor de la red de neuronas, con los pesos que habiamos guardado en el fit
        fAct_h = np.array([])
        for neurona in self.hide_w:
            temp = np.array([])
            for peso in range(1, self.hide_w[neurona], 1):
                temp = np.append(fAct_h, 1 / (1 + np.exp(- np.dot(neurona, x))))
            fAct_h = np.append(sum(temp))


        fAct_o = np.array([])
        for neurona in self.output_w:
            temp = np.array([])
            for peso in range(1, self.output_w[neurona], 1):
                temp = np.append(fAct_o, 1 / (1 + np.exp(- np.dot(neurona, fAct_h))))
            fAct_o = np.append(sum(temp))

        return fAct_o



class DeepMLP(object):
    def __init__(self, layers_size, learning_rate=0.):
        """
        ejemplos layers_size 
            [100, 50, 10] 100 neuronas de entrada, 50 de capa oculta 1 y 10 de salida
            [100, 50, 20, 10] 100 neuronas de entrada, 50 de capa oculta 1, 50 de capa oculta 2 y 10 de salida
            [100, 50, 20, 50, 10] etc.
        """
        aprendizaje=learning_rate
        size=len(layers_size)
        num_i=layers_size[0]
        num_o=layers_size[-1]
        num_h=np.array([])

        for i in range(1,size,1):
            num_h=np.append(num_h,layers_size[i])

        # Asignamos los pesos aleatorios para las distintas capas
        self.input_w = rand = np.random.uniform(-1, 1, size=num_i)
        self.output_w = rand = np.random.uniform(-1, 1, size=num_o)
        self.hide_w = rand = np.random.uniform(-1, 1, size=num_h)

        # Asignamos los pesos del bias de cada capa
        self.bias_o = rand = np.random.uniform(-1, 1, size=num_o)
        self.bias_h = rand = np.random.uniform(-1, 1, size=num_h)

    def fit(self, X, Y):
        """
        X = entradas del conjunto de datos de entrenamiento, puede ser un batch o una sola tupla
        Y = salidas esperadas del conjunto de datos de entrenamiento, puede ser un batch o una sola tupla
        """

        errorMAX = 1e-2
        sumaEO = errorMAX
        k = 0  # Variable que determinará cuantas iteracciones se han requerido para obtener el resultado
        done = False

        while not done:
            if sumaEO < errorMAX:  # Condición de parada del entrenamiento de la red
                done = True

            else:
                # Comienza el proceso de Feedfordward
                sumaEO = 0

                # Con este bucle se calcula por un lado  el sumatorio de todas las entradas de la neurona de la capa oculta por sus correspondientes pesos
                # para calcular la función activación de esta neurona, que en este caso será la sigmoidal
                for x, y in zip(self.X, self.Y):
                    suma=0


    def score(self, X, Y):
        """
        X = entradas del conjunto de datos de testeo, puede ser un batch o una sola tupla
        Y = salidas esperadas del conjunto de datos de testeo, puede ser un batch o una sola tupla
        """
        pass


if __name__ == '__main__':
    #from tensorflow.examples.tutorials.mnist import input_data
    #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #TODO MNIST TESTS
    #t=DeepMLP([1,2,3,4,5],0.5)

    # Pruebas para xorMLP
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    i = xorMLP(0.5)
    i.fit()
    for i in range(0, 4, 1):
        x = np.array(X[i])
        print("X:", x)
        print("Y esperada:", Y[i])
        res = int(i.predict(x))
        print("Y calculada: ", res, "\n")


