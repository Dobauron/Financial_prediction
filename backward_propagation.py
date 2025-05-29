import numpy as np


# forward
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prim(x):
    return sigmoid(x) * (1 - sigmoid(x))


x = np.array([1, 2, 3])

W1 = np.array(
    [
        [5, 4],
        [6, 3],
        [7, 3]
    ]
)

v_func = np.vectorize(sigmoid)
h_neurony_w_warstwie_ukrytej = np.matmul(x, W1)
wyjscie_z_warstwy_ukrytej = v_func(h_neurony_w_warstwie_ukrytej)

W2 = np.array(
    [1, 2]
)
h_neuron_wyjsciowy = np.dot(wyjscie_z_warstwy_ukrytej, W2)
wyjscie_z_sieci = sigmoid(h_neuron_wyjsciowy)
print(wyjscie_z_sieci)



# backward propagation
pozadane_wyjscie_z_sieci = -1
delta_neuronu_wyjsciowego = (pozadane_wyjscie_z_sieci - wyjscie_z_sieci) * sigmoid_prim(h_neuron_wyjsciowy)


delta_1_h = W2[0] * delta_neuronu_wyjsciowego * sigmoid_prim(h_neurony_w_warstwie_ukrytej[0])
delta_2_h = W2[1] * delta_neuronu_wyjsciowego * sigmoid_prim(h_neurony_w_warstwie_ukrytej[1])

dzeta = 0.1

delta_w_11 = dzeta * delta_1_h * x[0]
delta_w_21 = dzeta * delta_2_h * x[0]

delta_w_12 = dzeta * delta_1_h * x[1]
delta_w_22 = dzeta * delta_2_h * x[1]

delta_w_13 = dzeta * delta_1_h * x[2]
delta_w_23 = dzeta * delta_2_h * x[2]