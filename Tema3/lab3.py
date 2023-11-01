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

def is_unitary(matrix):
    return np.linalg.norm(matrix.dot(matrix.conj().T) - N * np.identity(N)) < 10e-10

print(is_unitary(F))

# Exercitiul 2

t = np.linspace(0, 1, 10000)
sinus = sinusoidal(t, 1, 4, np.pi / 2)
cerc_sinus = sinus * math.e ** -(2j * np.pi * t)

fig, axs = plt.subplots(6, figsize=(16, 16))
fig.subplots_adjust(hspace=0.5)

for i, ax in enumerate(axs):
    if i == 0:
        continue
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginar')
    ax.set_aspect('equal')
    ax.axhline(y=0, color='black')
    ax.axvline(x=0, color='black')

axs[0].set_xlabel('Timp')
axs[0].set_ylabel('Amplitudine')
axs[0].plot(t, sinus)
axs[0].scatter(t[3950], sinus[3950], color='red')
axs[0].plot([t[3950], t[3950]], [0, sinus[3950]], color='red')
axs[0].axhline(y=0, color='black')
ax.axvline(x=0, color='black')

axs[1].plot(np.real(cerc_sinus), np.imag(cerc_sinus))
axs[1].scatter(np.real(cerc_sinus[3950]), np.imag(cerc_sinus[3950]), color='red')
axs[1].plot([0, np.real(cerc_sinus[3950])], [0, np.imag(cerc_sinus[3950])], color='red')

def z(omega):
    return sinus * math.e ** -(2j * np.pi * omega * t)

z_1 = z(1)
z_4 = z(4)
z_5 = z(5)
z_10 = z(10)

axs[2].set_title('omega = 1')
axs[2].scatter(np.real(z_1), np.imag(z_1), c=np.abs(sinus), s=1)

axs[3].set_title('omega = 4')
axs[3].scatter(np.real(z_4), np.imag(z_4), c=np.abs(sinus), s=1)

axs[4].set_title('omega = 5')
axs[4].scatter(np.real(z_5), np.imag(z_5) ,c=np.abs(sinus), s=1)

axs[5].set_title('omega = 10')
axs[5].scatter(np.real(z_10), np.imag(z_10), c=np.abs(sinus), s=1)

fig.savefig('Tema3/ex_2.pdf')

# Exercitiul 3
fs = 4096
timp = np.linspace(0, 1, fs)
sinus_compus = sinusoidal(timp, 1, 192, np.pi / 2) + sinusoidal(timp, 2, 832, 0) + sinusoidal(timp, 0.5, 1600, 0) + sinusoidal(timp, 3, 32, 0)
N_ex3 = 128
X = np.zeros(N_ex3, dtype=np.complex_)

for i in range(N_ex3):
    for n in range(N_ex3):
        X[i] += sinus_compus[n] * math.e ** -(2j * np.pi * i * n / N_ex3)

fig, axs = plt.subplots(2, figsize=(16, 16))
fig.subplots_adjust(hspace=0.5)
axs[0].set_xlabel('Timp')
axs[0].set_ylabel('Amplitudine')
axs[0].plot(timp, sinus_compus)
axs[0].set_xlim(0, 0.1)
axs[0].set_title('Semnalul compus')
axs[0].axhline(y=0, color='black')

analized_frequencies = np.arange(0, fs, fs / N_ex3)
print(analized_frequencies)

axs[1].set_title('Transformata Fourier pentru un semnal cu 4 componente de frecventa')
axs[1].set_xlabel('Frecventa')
axs[1].set_ylabel('Modulul Transformatei Fourier')
axs[1].stem(analized_frequencies, np.abs(X))
# axs[1].stem(analized_frequencies[: N_ex3 // 2], np.abs(X[: N_ex3 // 2]))

fig.savefig('Tema3/ex_3.pdf')

