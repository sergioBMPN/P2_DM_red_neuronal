import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class xorMLP(object):
    def __init__(self, learning_rate=0.):
        self.aprendizaje = learning_rate
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.Y = np.array([[0], [1], [1], [0]])
        self.bias = -1

        ###################################################### Capa oculta 1 ################################################################
        rand = round(np.random.uniform(-1, 1), 5)
        rand2 = round(np.random.uniform(-1, 1), 5)
        self.capaOculta1 = np.array([rand, rand2])

        rand = round(np.random.uniform(-1, 1), 5)
        self.pesoBias_1_Oculta = np.array([rand])
        self.biasCapaOculta1 = self.bias

        #################################################### Capa oculta 2 ##################################################################
        rand = round(np.random.uniform(-1, 1), 5)
        rand2 = round(np.random.uniform(-1, 1), 5)
        self.capaOculta2 = np.array([rand, rand2])

        rand = round(np.random.uniform(-1, 1), 5)
        self.pesoBias_2_Oculta = np.array([rand])
        self.biasCapaOculta2 = self.bias

        ################################################### Capa salida #####################################################################
        rand = round(np.random.uniform(-1, 1), 5)
        rand2 = round(np.random.uniform(-1, 1), 5)
        self.capaSalida = np.array([rand, rand2])

        rand = round(np.random.uniform(-1, 1), 5)
        self.pesoBiasSalida = np.array([rand])
        self.biasCapaSalida = self.bias

    def fit(self):
        errorMAX = 0.10
        sumaEO = errorMAX
        k = 0
        done = False

        while not done:
            if sumaEO < errorMAX or k == 1000000:
                done = True

                self.pesosH1 = np.array(self.capaOculta1)
                self.pesoBiasH1 = np.array(self.pesoBias_1_Oculta)

                self.pesosH2 = np.array(self.capaOculta2)
                self.pesoBiasH2 = np.array(self.pesoBias_2_Oculta)

                self.pesosO = np.array(self.capaSalida)
                self.pesoBiasO = np.array(self.pesoBiasSalida)

            else:
                sumaEO = 0

                for x, y in zip(self.X, self.Y):

                    ###################################### Funcion activacion para oculta 1 #############################################
                    suma = self.biasCapaOculta1 * self.pesoBias_1_Oculta
                    for i in range(0, 2, 1):
                        suma += ((x[i] * self.capaOculta1[i]))

                    funcionActivacionOculta1 = 1 / (1 + np.exp(-suma))

                    ###################################### Funcion activacion para oculta 2 #############################################
                    suma = self.biasCapaOculta2 * self.pesoBias_2_Oculta
                    for i in range(0, 2, 1):
                        suma += ((x[i] * self.capaOculta2[i]))

                    funcionActivacionOculta2 = 1 / (1 + np.exp(-suma))

                    ###################################### Funcion activacion para salida ###############################################
                    suma = self.biasCapaSalida * self.pesoBiasSalida
                    suma += self.capaSalida[0] * funcionActivacionOculta1
                    suma += self.capaSalida[1] * funcionActivacionOculta2
                    funcionActivacionSalida = 1 / (1 + np.exp(-suma))

                    ###################################### Calculamos el error de salida ################################################
                    errorSalida = (funcionActivacionSalida * (1 - funcionActivacionSalida)) * (
                            y[0] - funcionActivacionSalida)

                    sumaEO += abs(errorSalida)

                    valorEntradaHO_1 = self.capaSalida[0]
                    valorEntradaHO_2 = self.capaSalida[1]

                    ##################################################incWho de la neurona oculta 1 #####################################
                    self.capaSalida[0] += self.aprendizaje * errorSalida * funcionActivacionOculta1

                    ##################################################incWho de la neurona oculta 2 #####################################
                    self.capaSalida[1] += self.aprendizaje * errorSalida * funcionActivacionOculta2

                    ################################################### error de la neurona oculta 1 ####################################
                    errorH_1 = (funcionActivacionOculta1 * (
                            1 - funcionActivacionOculta1)) * errorSalida * valorEntradaHO_1

                    ################################################### error de la neurona oculta 2 ####################################
                    errorH_2 = (funcionActivacionOculta2 * (
                            1 - funcionActivacionOculta2)) * errorSalida * valorEntradaHO_2

                    ################################################### incWih de la neurona oculta 1 ###################################
                    self.capaOculta1[0] += self.aprendizaje * errorH_1 * x[0]
                    self.capaOculta1[1] += self.aprendizaje * errorH_1 * x[1]
                    self.pesoBias_1_Oculta += self.aprendizaje * errorH_1 * self.biasCapaOculta1

                    ################################################### incWih de la neurona oculta 2 ###################################
                    self.capaOculta2[0] += self.aprendizaje * errorH_2 * x[0]
                    self.capaOculta2[1] += self.aprendizaje * errorH_2 * x[1]
                    self.pesoBias_2_Oculta += self.aprendizaje * errorH_2 * self.biasCapaOculta2

                    k += 1

        print("Iteraciones totales: ", k)

    def predict(self, x):
        # Una vez que hemos terminado de entrenar a la red de neuronas, para comprobar la precisión de la red,
        # en el predict, le pasamos como parámetro el array con las entradas y así obtener una salida estimada
        """
        x = [x1, x2]
        """
        # Proceso para obtener el valor de la red de neuronas, con los pesos que habiamos guardado en el fit
        suma = self.pesoBiasH1 * self.bias
        for i in range(0, 2, 1):
            suma += ((x[i] * self.pesosH1[i]))
        fActOculta1 = 1 / (1 + np.exp(-suma))

        suma = self.pesoBiasH2 * self.bias
        for i in range(0, 2, 1):
            suma += ((x[i] * self.pesosH2[i]))
        fActOculta2 = 1 / (1 + np.exp(-suma))

        suma = self.pesoBiasO * self.bias
        suma += self.pesosO[0] * fActOculta1
        suma += self.pesosO[1] * fActOculta2
        fActSalida = 1 / (1 + np.exp(-suma))

        # Devolvemos el valor de salida estimado
        return fActSalida


class DeepMLP(object):
    def __init__(self, layers_size, learning_rate=0.):
        """
        ejemplos layers_size
            [100, 50, 10] 100 neuronas de entrada, 50 de capa oculta 1 y 10 de salida
            [100, 50, 20, 10] 100 neuronas de entrada, 50 de capa oculta 1, 50 de capa oculta 2 y 10 de salida
            [100, 50, 20, 50, 10] etc.
        """

        self.aprendizaje = learning_rate
        self.tamanio_capas = layers_size
        self.pesosOculta = []
        self.pesosBias = []

        num_i = layers_size[0]
        num_o = layers_size[-1]

        self.X = tf.placeholder(tf.float32, [None, num_i])
        self.Y = tf.placeholder(tf.float32, [None, num_o])

        for num in range(0, len(layers_size) - 1, 1):
            rand = 2 * np.random.rand(layers_size[num], layers_size[num + 1]) - 1
            self.pesosOculta.append(tf.Variable(rand, dtype=tf.float32))

            rand = 2 * np.random.rand(1, layers_size[num + 1]) - 1
            self.pesosBias.append(tf.Variable(rand, dtype=tf.float32))

        self.sesion = tf.InteractiveSession()
        self.sesion.run(tf.global_variables_initializer())

    def fit(self, X):
        """
        X = entradas del conjunto de datos de entrenamiento, puede ser un batch o una sola tupla
        Y = salidas esperadas del conjunto de datos de entrenamiento, puede ser un batch o una sola tupla
        """
        # Feedfordward
        self.y = self.X
        for i in range(len(self.tamanio_capas) - 1):
            self.y = tf.matmul(self.y, self.pesosOculta[i]) + (self.pesosBias[i] * 1)

        # minimizamos el error optimizando el factor de aprendizaje
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=self.y))
        self.minimizar = tf.train.GradientDescentOptimizer(self.aprendizaje).minimize(cross_entropy)

        done = False
        k = 0  # contador de vueltas
        x_train, y_train = X.next_batch(100)
        self.score(x_train, y_train)
        while done == False:
            k += 1
            if k >= 100000:
                done = True

            elif k % 1000 == 0:  # cada 100 iteraciones se hace una actualizacion de la precision del entrenamiento
                train_acc = self.accuracy.eval(feed_dict={self.X: x_train, self.Y: y_train})
                if (train_acc >= 0.98):
                    done = True
                else:
                    print("Iteraciones:" + str(k) + ", Acc: " + str(train_acc))

            # actualizamos el factor de aprendizaje
            self.sesion.run(self.minimizar, feed_dict={self.X: x_train, self.Y: y_train})
            # cogemos el sugiente batch de entrenamiento
            x_train, y_train = X.next_batch(100)

        train_acc = self.accuracy.eval(feed_dict={self.X: x_train, self.Y: y_train})
        print("Iteraciones:" + str(k) + ", Acc: " + str(train_acc))

    def score(self, X, Y):
        """
        X = entradas del conjunto de datos de testeo, puede ser un batch o una sola tupla
        Y = salidas esperadas del conjunto de datos de testeo, puede ser un batch o una sola tupla
        """
        # comparamos la Y                estimada            con la real
        compare = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(compare, tf.float32))
        self.a = self.sesion.run(self.accuracy, feed_dict={self.X: X, self.Y: Y})
        print('Acc %g' % self.a)


if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Pruebas para xorMLP

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    print("XOR 2-2-1:")
    red = xorMLP(0.2)
    red.fit()

    for i in range(0, 4, 1):
        x = np.array(X[i])
        print("X:", x)
        print("Y:", Y[i])
        res = red.predict(x)
        print("Y de la red: ", res)
        print("Y esperada: ", round(res[0], 0))

    print("\n")
    print("Tensor:")

    deepmlp = DeepMLP([784, 10, 10, 10], 0.2)
    X_train = mnist.train
    deepmlp.fit(X_train)
    X_test = mnist.test.images
    Y_test = mnist.test.labels
    deepmlp.score(X_test, Y_test)