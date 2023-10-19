import numpy as np
from matplotlib import pyplot as plt
import sounddevice
import scipy

# Exercitiul 1
def sinusoidal(t, A, f, phi):
    return A * np.sin(2 * np.pi * f * t + phi)
def cosinusoidal(t, A, f, phi):
    return A * np.cos(2 * np.pi * f * t + phi)

t_ex1 = np.linspace(0, np.pi, 10000)

fig, axs = plt.subplots(2)
for ax in axs.flat:
    ax.set_xlabel('Timp')
    ax.set_ylabel('Amplitudine')
fig.suptitle('Exercitiul 1')

axs[0].set_title('Semnal sinusoidal (phi = 0)')
axs[0].plot(t_ex1, sinusoidal(t_ex1, 1, 1 / np.pi, 0))

axs[1].set_title('Semnal cosinusoidal (phi = 3 * pi / 2)')
axs[1].plot(t_ex1, cosinusoidal(t_ex1, 1, 1 / np.pi, 3 * np.pi / 2))

fig.savefig('Tema2/ex_1.pdf')

# Exercitiul 2
t_ex2 = np.linspace(0, 1, 10000)
sin_1 = sinusoidal(t_ex2, 1, 100, 0)
sin_2 = sinusoidal(t_ex2, 1, 100, 1)
sin_3 = sinusoidal(t_ex2, 1, 100, 2)
sin_4 = sinusoidal(t_ex2, 1, 100, 3)

z = np.random.normal(size=10000)
gamma_1 = np.sqrt(np.linalg.norm(sin_1) / (0.1 * np.linalg.norm(z)))
gamma_2 = np.sqrt(np.linalg.norm(sin_2) / (1 * np.linalg.norm(z)))
gamma_3 = np.sqrt(np.linalg.norm(sin_3) / (10 * np.linalg.norm(z)))
gamma_4 = np.sqrt(np.linalg.norm(sin_4) / (100 * np.linalg.norm(z)))

plt.figure(123)
plt.xlim((0, 0.1))

plt.plot(t_ex1, sin_1 + gamma_1 * z, label='phi = 0; snr = 0.1')
plt.plot(t_ex1, sin_2 + gamma_2 * z, label='phi = 1; snr = 1')
plt.plot(t_ex1, sin_3 + gamma_3 * z, label='phi = 2; snr = 10')
plt.plot(t_ex1, sin_4 + gamma_4 * z, label='phi = 3; snr = 100')
plt.legend()

plt.savefig('Tema2/ex_2.pdf')

# Exercitiul 3
def sawtooth(t, f):
    return f * t - np.floor(f * t)

f_400 = 400
nr_esantioane = 1600
timp = np.linspace(0, 0.1, nr_esantioane)
semnal_sinusoidal_400 = np.sin(2 * np.pi * f_400 * timp)
# sounddevice.play(semnal_sinusoidal_400, 44100)
f_800 = 800
t_3s = np.linspace(0, 3, 100000)
semnal_sinusoidal_800 = np.sin(2 * np.pi * f_800 * t_3s)
# sounddevice.play(semnal_sinusoidal_800, 44100)
f_240 = 240
t_sawtooth = np.linspace(0, 0.1, 10000)
semnal_sawtooth_240 = f_240 * t_sawtooth - np.floor(f_240 * t_sawtooth)
# sounddevice.play(semnal_sawtooth_240, 44100)
f_300 = 300
t_square = np.linspace(0, 1, 10000)
semnal_square_300 = np.sign(np.sin(2 * np.pi * f_300 * t_square))
# sounddevice.play(semnal_square_300, 44100)
rate = int(10e5)
scipy.io.wavfile.write('Tema2/square300.wav', rate, semnal_square_300)

rate, x = scipy.io.wavfile.read('Tema2/square300.wav')
sounddevice.play(x, 44100)

# Exercitiul 4
t_ex4 = np.linspace(0, 3, 10000)

fig, axs = plt.subplots(3)
plt.subplots_adjust(hspace=1)
for ax in axs.flat:
    ax.set_xlabel('Timp')
    ax.set_ylabel('Amplitudine')
fig.suptitle('Exercitiul 4')

axs[0].set_title('Semnal sinusoidal ')
axs[0].plot(t_ex4, sinusoidal(t_ex4, 1, 1 / np.pi, 0))

axs[1].set_title('Semnal sawtooth')
axs[1].plot(t_ex4, sawtooth(t_ex4, 1))

axs[2].set_title('Semnal sinusoidal + semnal sawtooth')
axs[2].plot(t_ex4, sinusoidal(t_ex4, 1, 1 / np.pi, 0) + sawtooth(t_ex4, 1))

fig.savefig('Tema2/ex_4.pdf')
