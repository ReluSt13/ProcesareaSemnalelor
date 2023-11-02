import numpy as np
from matplotlib import pyplot as plt
import math
from timeit import default_timer as timer

def sinusoidal(t, A, f, phi):
    return A * np.sin(2 * np.pi * f * t + phi)

# Exercitiul 1

fs = 8192 * 4
timp = np.linspace(0, 1, fs)
sinus_compus = sinusoidal(timp, 1, 192, np.pi / 2) + sinusoidal(timp, 2, 832, 0) + sinusoidal(timp, 0.5, 1600, 0) + sinusoidal(timp, 3, 32, 0)
N = [128, 256, 512, 1024, 2048, 4096, 8192]
t_fft_manual = np.zeros(len(N))
t_fft_numpy = np.zeros(len(N))

for i in range(len(N)):
    X = np.zeros(N[i], dtype=np.complex_)
    start = timer()
    for omega in range(N[i]):
        for n in range(N[i]):
            X[omega] += sinus_compus[n] * math.e ** -(2j * np.pi * omega * n / N[i])
    end = timer()
    t_fft_manual[i] = end - start
print(t_fft_manual)

for i in range(len(N)):
    start = timer()
    X = np.fft.fft(sinus_compus, N[i])
    end = timer()
    t_fft_numpy[i] = end - start
    print(end - start)

print(t_fft_numpy)

plt.figure(0)
plt.xlabel('N')
plt.ylabel('Timp')
plt.plot(N, t_fft_manual, label='Manual')
plt.plot(N, t_fft_numpy, label='Numpy')
plt.yscale('log')
plt.title('Timpul de executie al transformatei Fourier')
plt.legend()
plt.grid()

plt.savefig('Tema4/ex_1.pdf')

# Exercitiul 2
t_ex2 = np.linspace(0, 1, 150) # 150 < 2 * 100 (100 frecventa sinusului)
sinus = sinusoidal(t_ex2, 1, 100, 0)
fig, axs = plt.subplots(3, figsize=(16, 16))
fig.subplots_adjust(hspace=0.5)
axs[0].set_xlabel('Timp')
axs[0].set_ylabel('Amplitudine')
axs[0].plot(t_ex2, sinus)
axs[0].stem(t_ex2, sinus)
axs[0].set_xlim(0, 0.1)
fig.savefig('Tema4/ex_2.pdf')

