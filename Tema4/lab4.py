import numpy as np
from matplotlib import pyplot as plt
import math
from timeit import default_timer as timer
import sounddevice
import scipy

def sinusoidal(t, A, f, phi):
    return A * np.sin(2 * np.pi * f * t + phi)

Exercitiul 1

fs = 10000
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
t_afis = np.linspace(0, 1, 30000)
fs = 150
t_ex2 = np.linspace(0, 1, fs + 1)

f0 = 100
sinus = sinusoidal(t_ex2, 1, f0, 0)
sinus_afis_1 = sinusoidal(t_afis, 1, f0, 0)
# fs < 2 * f0 (sub nyquist)
# fs = 150; f0 = 100; ==> pentru fo + k * fs voi folosi k = 1 si k = -2
sinus_2 = sinusoidal(t_ex2, 1, f0 + 1 * fs, 0)
sinus_afis_2 = sinusoidal(t_afis, 1, f0 + 1 * fs, 0)

sinus_3 = sinusoidal(t_ex2, 1, f0 + -2 * fs, 0)
sinus_afis_3 = sinusoidal(t_afis, 1, f0 + -2 * fs, 0)

fig, axs = plt.subplots(3, figsize=(16, 16))
fig.subplots_adjust(hspace=0.5)
for i, ax in enumerate(axs):
    ax.set_xlabel('Timp')
    ax.set_ylabel('Amplitudine')
    ax.set_xlim(0, 0.1)

axs[0].plot(t_afis, sinus_afis_1)
axs[0].stem(t_ex2, sinus)

axs[1].stem(t_ex2, sinus_2)
axs[1].plot(t_afis, sinus_afis_2)

axs[2].stem(t_ex2, sinus_3)
axs[2].plot(t_afis, sinus_afis_3)

print(np.linalg.norm(sinus - sinus_2) < 1e-10)
print(np.linalg.norm(sinus - sinus_3) < 1e-10)

fig.savefig('Tema4/ex_2.pdf')

# Exercitiul 3

# fs_ex3 > 2 * f0 ( > frecv. nyquist) --> 500 > 2 * 100
fs_ex3 = 500
t_ex3 = np.linspace(0, 1, fs_ex3)
sinus_ex3 = sinusoidal(t_ex3, 1, f0, 0)

sinus_2_ex3 = sinusoidal(t_ex3, 1, f0 + 1 * fs_ex3, 0)
sinus_afis_2_ex3 = sinusoidal(t_afis, 1, f0 + 1 * fs_ex3, 0)

sinus_3_ex3 = sinusoidal(t_ex3, 1, f0 + -2 * fs_ex3, 0)
sinus_afis_3_ex3 = sinusoidal(t_afis, 1, f0 + -2 * fs_ex3, 0)

fig, axs = plt.subplots(3, figsize=(16, 16))
fig.subplots_adjust(hspace=0.5)
for i, ax in enumerate(axs):
    ax.set_xlabel('Timp')
    ax.set_ylabel('Amplitudine')
    ax.set_xlim(0, 0.03)
axs[0].stem(t_ex3, sinus_ex3)
axs[0].plot(t_afis, sinus_afis_1)

axs[1].stem(t_ex3, sinus_2_ex3)
axs[1].plot(t_afis, sinus_afis_2_ex3)

axs[2].stem(t_ex3, sinus_3_ex3)
axs[2].plot(t_afis, sinus_afis_3_ex3)

print(np.linalg.norm(sinus_ex3 - sinus_2_ex3) < 1e-10)
print(np.linalg.norm(sinus_ex3 - sinus_3_ex3) < 1e-10)

fig.savefig('Tema4/ex_3.pdf')

# Exercitiul 4
""" Contrabas ---> 40Hz - 200Hz 
    frecventa maxima = 200Hz
    ca semnalul discretizat sa contina toate componentele de frecventa pe care instrumentul le poate produce,
    trebuie sa avem frecventa de esantionare fs > 2 * 200Hz = 400Hz (cel putin)
"""

# Exercitiul 5 -> sunt destul de similare... u pare sa fie putin mai diferit de restul

# Exercitiul 6
rate, data = scipy.io.wavfile.read('Tema4/vocale_2_mono.wav')

marime_grup = int(len(data) * 0.01)
marime_suprapunere = marime_grup // 2

spectograma = []
for i in range(0, len(data) - marime_grup, marime_suprapunere):
    grup = data[i:i + marime_grup]
    fft_grup = np.abs(np.fft.fft(grup))
    spectograma.append(fft_grup[:len(fft_grup) // 2])

# convertesc la decibeli relativ la valoarea maxima
spectograma_db = 20 * np.log10(np.array(spectograma).T / np.max(spectograma)) # folosesc transpusa matricei deoarece am adaugat fft-urile pe linii intial

threshold_db = -100

spectograma_db[spectograma_db < threshold_db] = threshold_db

plt.figure(100)
plt.imshow(spectograma_db, aspect='auto', origin='lower', extent=[0, len(data) / rate, 0, rate / 2])
plt.colorbar(label='Magnitudine (dB)')
plt.xlabel('Timp (s)')
plt.ylabel('Frecventa (Hz)')
plt.savefig('Tema4/ex_6.pdf')

# Exercitiul 7
"""
    Psemnal = 90dB
    SNRdB = 80dB
    Pzgomot = ?
    SNRdB = 10 * log10(SNR)
    ==> SNR = 10 ^ 8  ===> (10 * 8 = 80dB)
    SNR = Psemnal / Pzgomot
    Psemnal = 90dB = 10 ^ 9
    Pzgomot = 10 ^ 9 / 10 ^ 8 = 10 = 10dB
"""

