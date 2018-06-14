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
    a=np.array([1,1,2])
    b=np.array([2,2,3])
    c=np.array([a[0]])
    temp = np.array(np.concatenate(([0], a)))
    print(sum([1]))