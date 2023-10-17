import numpy as np
import matplotlib.pyplot as plt

# Exercitiul 1

def x(t):
    return np.cos(520 * np.pi * t + (np.pi / 3))

def y(t):
    return np.cos(280 * np.pi * t - (np.pi / 3))

def z(t):
    return np.cos(120 * np.pi * t + (np.pi / 3))

fig, axs = plt.subplots(6, figsize=(6, 10))
plt.subplots_adjust(hspace=0.5)
fig.suptitle('Exercitiul 1')
evenly_spaced_numbers = np.linspace(0, 0.03, 60)
x_fun_values = x(evenly_spaced_numbers)
y_fun_values = y(evenly_spaced_numbers)
z_fun_values = z(evenly_spaced_numbers)
axs[0].plot(evenly_spaced_numbers, x_fun_values)
axs[1].plot(evenly_spaced_numbers, y_fun_values)
axs[2].plot(evenly_spaced_numbers, z_fun_values)

#c
t = np.linspace(0, 1, 200)
t_more_intervals = np.linspace(0, 1, 10000) # pt afisarea semnalului 'continuu' deasupra semnalului discret
valori_x = x(t)
valori_y = y(t)
valori_z = z(t)
valori_cont_x = x(t_more_intervals)
valori_cont_y = y(t_more_intervals)
valori_cont_z = z(t_more_intervals)

for index, ax in enumerate(axs.flat):
    ax.set_xlabel('Timp')
    ax.set_ylabel('Amplitudine')
    if index >= 3:
        ax.set_xlim([0, 0.06])
axs[3].plot(t_more_intervals, valori_cont_x)
axs[3].stem(t, valori_x)
axs[4].plot(t_more_intervals, valori_cont_y)
axs[4].stem(t, valori_y)
axs[5].plot(t_more_intervals, valori_cont_z)
axs[5].stem(t, valori_z)


fig.savefig('Lab1/ex_1.pdf')

# Exercitiul 2

f_400 = 400
nr_esantioane = 1600

timp = np.linspace(0, 0.1, nr_esantioane)
semnal_sinusoidal_400 = np.sin(2 * np.pi * f_400 * timp)

fig, axs = plt.subplots(4, figsize=(10, 10))
plt.subplots_adjust(hspace=1.5)
fig.suptitle('Exercitiul 2')
for ax in axs.flat:
    ax.set_xlabel('Timp')
    ax.set_ylabel('Amplitudine')
axs[0].set_title('Semnal sinusoidal de frecventa 400 Hz')
axs[0].set_xlim([0, 0.01])
axs[0].plot(timp, semnal_sinusoidal_400)

f_800 = 800
t_3s = np.linspace(0, 3, 100000)
semnal_sinusoidal_800 = np.sin(2 * np.pi * f_800 * t_3s)

axs[1].set_title('Semnal sinusoidal de frecventa 800 Hz')
axs[1].set_xlim([0, 0.01])
axs[1].plot(t_3s, semnal_sinusoidal_800)

f_240 = 240
t_sawtooth = np.linspace(0, 0.1, 10000)
semnal_sawtooth_240 = f_240 * t_sawtooth - np.floor(f_240 * t_sawtooth)

axs[2].set_title('Semnal sawtooth de frecventa 240 Hz')
axs[2].set_xlim([0, 0.025])
axs[2].plot(t_sawtooth, semnal_sawtooth_240)

f_300 = 300
t_square = np.linspace(0, 0.1, 10000)
semnal_square_300 = np.sign(np.sin(2 * np.pi * f_300 * t_square))

axs[3].set_title('Semnal square de frecventa 300 Hz')
axs[3].set_xlim([0, 0.02])
axs[3].plot(t_square, semnal_square_300)

fig.savefig('Lab1/ex_2.pdf')

random_array = np.random.rand(128, 128)
plt.figure(542)
plt.title('Semnal de marime 128 * 128 random')
plt.imshow(random_array)
plt.colorbar()
plt.savefig('Lab1/ex_2_rand.pdf')


arr = np.zeros((128, 128))
for i in range(128):
    for j in range(128):
        arr[i, j] = np.abs(i - j)
plt.figure(545)
plt.title('Some gradient signal')
plt.imshow(arr)
plt.colorbar()
plt.savefig('Lab1/ex_2_gradient.pdf')

"""
    Exercitiul 3
    frecv_esantionare = 2000Hz
    a) intervalul intre doua esantioane este 1 / 2000 = 0.0005secunde sau 500 microsecunde
    b) 1 esantion ... 4 biti
       
       2000 esantioane .... 1 secunda   |
       x esantioane .......3600 secunde |   -> x = 7200000 esantioane intr-o ora

       1 esantion          ...  4 biti  |
       7200000 esantioane  ...  x biti  |   -> 28800000 biti * 0.125 = 3600000 bytes = 3600 kilobytes = 3.6 megabytes
"""


