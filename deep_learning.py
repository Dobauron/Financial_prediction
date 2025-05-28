import numpy as np

# x = np.array([3, -5])
# W = np.array([2, -3]).T #transpozycja - odwrócenie wektora z wierszowego na kolumnowy
# #ustawienie wektoru na kolumnowy inaczej źle sie wymanarza
#
# suma_wazona = np.dot(x, W)
# print(suma_wazona)


def reLU(x):
    if x > 0:
        return x
    else:
        return 0
x = np.array([1,2])

w11 = 6
w21 = 4

w12 = 7
w22 = 5

w13 = 9
w23 = 8

W = np.array(
    [
        [w11,w12,w13],
        [w21,w22,w23]

    ]
)

suma_wazona = np.matmul(x, W)

v_func = np.vectorize(reLU)
wyjscie_z_sieci = v_func(suma_wazona)
print(wyjscie_z_sieci)