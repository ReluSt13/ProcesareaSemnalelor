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
fig.subplots_adjust(hspace=0.5)
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
# sounddevice.play(x, 44100)

# Exercitiul 4
t_ex4 = np.linspace(0, 3, 10000)

fig, axs = plt.subplots(3)
fig.subplots_adjust(hspace=1)
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

# Exercitiul 5
"""
I created two one second signals, the first being middle C and the second being middle G.
When the signals are appended and then played one can hear the C for one second followed by the G for another second.
""" 
t_ex5 = np.linspace(0, 1, 44100) # results in 1 second sound --> duration = number of samples / sample rate
sawtooth_vals_middle_c = sawtooth(t_ex5, 261.63)
# sounddevice.play(sawtooth_vals_middle_c, 44100) commented so that i dont hear the audio every time i run this
# sounddevice.wait()
sawtooth_vals_middle_g = sawtooth(t_ex5, 392)
# sounddevice.play(sawtooth_vals_middle_g, 44100)
# sounddevice.wait()
sawtooth_vals_combined = np.append(sawtooth_vals_middle_c, sawtooth_vals_middle_g)
# sounddevice.play(sawtooth_vals_combined, 44100)
# sounddevice.wait()

# Exercitiul 6
"""
    The first wave with f = fs / 2 almost appears as a straight line.
    This is happening because there are only two samples per cycle and these samples are taken very close to zero-crossings; this leads to very poor resolution.
    The second wave with f = fs / 4 resembles more a sinusoidal wave than the first because now there are four samples per cycle leading to a higher resolution.
    Yet the signal doesn't look 'curved' like the sin wave when plotted because there are only 4 points per wave-length and matplotlib just connects them using a straight line.
    The last signal with f = 0 Hz resembles the first when plotted because this one only consists of values of 0.
"""
t_ex6 = np.linspace(0, 1, 1000)
sinus_1 = sinusoidal(t_ex6, 1, 500, 0)
sinus_2 = sinusoidal(t_ex6, 1, 250, 0)
sinus_3 = sinusoidal(t_ex6, 1, 0, 0)
t_show = np.linspace(0, 1, 50000)
sinus_1_show = sinusoidal(t_show, 1, 500, 0)
sinus_2_show = sinusoidal(t_show, 1, 250, 0)

fig, axs = plt.subplots(3)
fig.subplots_adjust(hspace=1)
for ax in axs.flat:
    ax.set_xlabel('Timp')
    ax.set_ylabel('Amplitudine')
    ax.set_xlim([0, 0.01])
fig.suptitle('Exercitiul 6')

axs[0].set_title('Semnal sinusoidal f = fs / 2')
axs[0].plot(t_show, sinus_1_show)
axs[0].plot(t_ex6, sinus_1)
axs[0].stem(t_ex6, sinus_1)

axs[1].set_title('Semnal sinusoidal f = fs / 4')
axs[1].plot(t_show, sinus_2_show)
axs[1].plot(t_ex6, sinus_2)
axs[1].stem(t_ex6, sinus_2)

axs[2].set_title('Semnal sinusoidal f = 0 Hz')
axs[2].plot(t_ex6, sinus_3)
axs[2].stem(t_ex6, sinus_3)

fig.savefig('Tema2/ex_6.pdf')

# Exercitiul 7
"""
    One of the differences between these signals is the resolution. The first one has 1000 sample points while the other two only have 250.
    The more the frequency increases the worse the signal will appear because will be less samples per wave-length. 
    The second signal when compared to the first starts 'a bit later' since we begin from the fourth value.
    The thrid signal also starts 'a bit later' since we begin from the second value but its not as visible as the second signal.
    Nonetheless, at lower sin frequencies, the signals look very similar. We start to see more evident differences when raising the frequency.
    At higher frequencies, the two decimated signals look similar but they are shifted.
"""
t_ex7 = np.linspace(0, 1, 1000)
sin_ex7 = sinusoidal(t_ex7, 1, 100, 0)

t_decimat_4 = t_ex7[3::4]
sin_decimat_4 = sin_ex7[3::4]

t_decimat_2 = t_ex7[1::4]
sin_decimat_2 = sin_ex7[1::4]

fig, axs = plt.subplots(3)
fig.subplots_adjust(hspace=1)
for ax in axs.flat:
    ax.set_xlabel('Timp')
    ax.set_ylabel('Amplitudine')
    ax.set_xlim([0, 0.1])
fig.suptitle('Exercitiul 7')

axs[0].set_title('Semnal sinusoidal')
axs[0].plot(t_ex7, sin_ex7)

axs[1].set_title('Semnal sinusoidal decimat pornind de la al 4-lea element')
axs[1].plot(t_decimat_4, sin_decimat_4)

axs[2].set_title('Semnal sinusoidal decimat pornind de la al 2-lea element')
axs[2].plot(t_decimat_2, sin_decimat_2)

fig.savefig('Tema2/ex_7.pdf')

# Exercitiul 8

interval = np.linspace(-np.pi / 2, np.pi / 2, 10000)
approx_pade = (interval - (7 / 60) * interval ** 3) / (1 + (interval ** 2) / 20)
fig, axs = plt.subplots(6, figsize=(8, 8))
fig.subplots_adjust(hspace=1.5)
for index, ax in enumerate(axs.flat):
    ax.set_xlabel('Timp')
    ax.set_ylabel('Amplitudine')
    if index % 2 == 1 or index >= 4:
        ax.set_xlabel('a')
        ax.set_ylabel('Eroare')
fig.suptitle('Exercitiul 8')

axs[0].set_title('sin(a) vs taylor')
axs[0].plot(interval, interval, label='a')
axs[0].plot(interval, np.sin(interval), label='sin(a)')
axs[0].legend()

axs[1].set_title('Eroare Taylor')
axs[1].plot(interval, np.abs(np.sin(interval) - interval))

axs[2].set_title('sin(a) vs pade')
axs[2].plot(interval, approx_pade, label='Pade')
axs[2].plot(interval, np.sin(interval), label='sin(a)')
axs[2].legend()

axs[3].set_title('Eroare Pade')
axs[3].plot(interval, np.abs(np.sin(interval) - approx_pade))


axs[4].set_title('Eroare Taylor (axa 0y logaritmica)')
axs[4].plot(interval, np.abs(np.sin(interval) - interval))
axs[4].set_yscale('log')

axs[5].set_title('Eroare Pade (axa 0y logaritmica)')
axs[5].plot(interval, np.abs(np.sin(interval) - approx_pade))
axs[5].set_yscale('log')

fig.savefig('Tema2/ex_8.pdf')
