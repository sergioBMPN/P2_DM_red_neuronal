import numpy as np

def sigmoid(x, derivative=False):
    if (derivative == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def funct_act(x,bias, peso, num_entradas):
    suma = peso * bias
    for i in range(0, num_entradas, 1):
        suma += ((x[i] * peso[i]))
    fActOculta = 1 / (1 + np.exp(-suma))

if __name__ == '__main__':
    rand = np.random.uniform(-1, 1, size=8)
    rand = rand.round(5)
    print(np.array([(rand[0], rand[1]),(rand[2], rand[3])]))