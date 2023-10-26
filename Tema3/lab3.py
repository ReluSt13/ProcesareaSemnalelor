import numpy as np
from matplotlib import pyplot as plt
import math

def sinusoidal(t, A, f, phi):
    return A * np.sin(2 * np.pi * f * t + phi)

# Exercitiul 1

N = 8
F = np.ones((8, 8), dtype=np.complex_)
for i in range(1, 8):
    for j in range(1, 8):
        F[i, j] = math.e ** (2j * np.pi * i * j / N)

real_part = np.real(F)
imag_part = np.imag(F)

fig, axs = plt.subplots(8)
for i in range(8):
    axs[i].plot(np.arange(8), real_part[i])
    axs[i].plot(np.arange(8), imag_part[i], linestyle='dashed')
fig.savefig('Tema3/ex_1.pdf')

print(np.linalg.norm(F.dot(F.conj().T) - N * np.identity(8)) < 10e-10)

# Exercitiul 2

t = np.linspace(0, 1, 10000)
sinus = sinusoidal(t, 1, 8, 0)
cerc_sinus = sinus * math.e ** -(2j * np.pi * t)


fig, axs = plt.subplots(2)
fig.subplots_adjust(hspace=0.5)
axs[0].set_xlabel('Timp')
axs[0].set_ylabel('Amplitudine')
axs[0].plot(t, sinus)

axs[1].set_xlabel('Real')
axs[1].set_ylabel('Imaginar')
axs[1].set_aspect('equal')
axs[1].plot(np.real(cerc_sinus), np.imag(cerc_sinus))
fig.savefig('Tema3/ex_2.pdf')