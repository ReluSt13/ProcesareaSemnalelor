import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

# ex1)
# Cerinta: Generati un vector x[n] aleator de dimensiune N = 100. Calculati iteratia x <- x * x (convolutie) de trei ori. Afisati cele patru grafice
N = 100
x = np.random.rand(N)
x1 = np.convolve(x, x)
x2 = np.convolve(x1, x)
x3 = np.convolve(x2, x)
fig, axs = plt.subplots(4)
fig.subplots_adjust(hspace=0.5)
fig.suptitle('Convolutie')
axs[0].plot(x)
axs[1].plot(x1)
axs[2].plot(x2)
axs[3].plot(x3)
fig.savefig('Tema6/ex1.pdf')
# Ce observati?
# se face blending cu o versiune 'intarziata' a lui

# ex2)
# Cerinta: Vi se dau doua polinoame p(x) si q(x) cu grad maxim N generate aleator cu coeficienti intregi.
# Calculati produsul lor r(x) = p(x)q(x) folosind convolutia: folosind inmultirea polinoamelor directa si apoi folosind fft.
def multiply_polynomials(p, q):
    r = np.zeros(len(p) + len(q) - 1)
    for i in range(len(p)):
        for j in range(len(q)):
            r[i + j] += p[i] * q[j]
    return r

N = 10
p = np.random.randint(10, size=N)
q = np.random.randint(10, size=N)
r_de_mana = multiply_polynomials(p, q)
r_np_convolve = np.convolve(p, q)
print(r_de_mana)
print(r_np_convolve)

p_fft = np.fft.fft(p)
q_fft = np.fft.fft(q)
r_fft = np.fft.ifft(p_fft * q_fft)
print(r_fft.real) # i only get 10 coefficients, not 19
# solution: pad the vectors with zeros
p_padded = np.pad(p, (0, len(p) - 1), 'constant')
q_padded = np.pad(q, (0, len(q)  -1), 'constant')
p_fft = np.fft.fft(p_padded)
q_fft = np.fft.fft(q_padded)
r_fft = np.fft.ifft(p_fft * q_fft)
print(r_fft.real)

# ex3)
# Cerinta: Scrieti cate o functie prin care sa construiti o fereastra dreptunghiulara si o fereastra de tip Hanning.
# Functiile primesc ca parametru dimensiunea ferestrei. Afisati grafic o sinusoidala cu f = 100, A = 1 si phi = 0 trecuta prin cele doua tipuri de ferestre de dimensiune N_w = 200
def rectangular_window(N_w):
    return np.ones(N_w)
def hanning_window(N_w):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(N_w) / N_w))
def sinusoidal(t, A, f, phi):
    return A * np.sin(2 * np.pi * f * t + phi)

t = np.linspace(0, 1, 10000)
x = sinusoidal(t, 1, 100, 0)
N_w = 200
w1 = rectangular_window(N_w)
w2 = hanning_window(N_w)
fig, axs = plt.subplots(3)
fig.subplots_adjust(hspace=1)
fig.suptitle('Ferestre')
for ax in axs:
    ax.set_xlim(0.48, 0.52)
    ax.set_xlabel('Timp')
    ax.set_ylabel('Amplitudine')
# pad w1 and w2 with zeros left and right
w1 = np.pad(w1, (len(x) - len(w1)) // 2, 'constant')
w2 = np.pad(w2, (len(x) - len(w2)) // 2, 'constant')
axs[0].plot(t, x)
axs[1].set_title('Fereastra dreptunghiulara')
axs[1].plot(t, x * w1)
axs[2].set_title('Fereastra Hanning')
axs[2].plot(t, x * w2)
fig.savefig('Tema6/ex3.pdf')

# ex4)
# Fisierul Train.csv contine date de trafic inregistrate pe o perioada de 1 saptamana.
# Perioada de esantionare este de 1 ora, iar valorile masurate reprezinta numarul de vehicule ce trec printr-o anumita locatie.

# Selectati din semnalul dat o portiune corespunzatoare pentru 3 zile, x, pe care veti lucra in continuare.
data = np.genfromtxt('Tema6/Train.csv', delimiter=',', skip_header=1, dtype=None, encoding=None)
x = data[(len(data) // 2 - 36):(len(data) // 2 + 36)]
x_count_data = np.array([int(entry[2]) for entry in x])
# Utilizati functia np.convolve(x, np.ones(w), 'valid) / w
# pentru a realiza un filtru de tip medie alunecatoare si neteziti semnalul obtinut anterior.
# Setati dimensiuni diferite ale ferestrei (variabila w in codul de mai sus), spre exemplu 5, 9, 13, 17
x_filtered_w5 = np.convolve(x_count_data, np.ones(5), 'valid') / 5
x_filtered_w9 = np.convolve(x_count_data, np.ones(9), 'valid') / 9
x_filtered_w13 = np.convolve(x_count_data, np.ones(13), 'valid') / 13
x_filtered_w17 = np.convolve(x_count_data, np.ones(17), 'valid') / 17

plt.figure(12345)
plt.plot(x_count_data, label='original')
plt.plot(x_filtered_w5, label='w=5')
plt.plot(x_filtered_w9, label='w=9')
plt.plot(x_filtered_w13, label='w=13')
plt.plot(x_filtered_w17, label='w=17')
plt.legend()
plt.savefig('Tema6/ex4_b.pdf')

# Doriti sa filtrati zgomotul (frecvente inalte) din semnalul cu date de trafic; alegeti o frecventa de taiere pentru un filtru trece-jos
# pe care il veti crea in continuare. Argumentati. Care este valoarea frecventei in Hz si care este valoarea frecventei normalizate
# intre 0 si 1 unde 1 reprezinta frecventa Nyquist?

# Frecventa de esantionare este de 1 / 3600 Hz
# Frecventa maxima este de 1 / 7200 Hz (frecventa Nyquist)
# Aleg frecventa inalta tot de 1 / 3600 Hz (frecventa de taiere)
# Frecventa normalizata este de 1 / 2 = 0.5

# Utilizand functiile si scipy.signal.butter si scipy.signal.cheby1 proiectati filtrele Butterworth si Chebyshev de ordin 5,
# cu frecventa de taiere stabilita mai sus. Pentru inceput setati atenuarea ondulatiilor, rp = 5 dB, urmand ca apoi sa incercati si alte valori.
coef_a_butter, coef_b_butter = signal.butter(5, 0.5, 'low')
coef_a_cheby, coef_b_cheby = signal.cheby1(5, 5, 0.5, 'low')
filtered_with_butter = signal.filtfilt(coef_a_butter, coef_b_butter, x_count_data)
filtered_with_cheby = signal.filtfilt(coef_a_cheby, coef_b_cheby, x_count_data)
plt.figure(123456)
plt.plot(x_count_data, label='original')
plt.plot(filtered_with_butter, label='butter')
plt.plot(filtered_with_cheby, label='cheby')
plt.legend()
plt.savefig('Tema6/ex4_d.pdf')

# Ce filtru alegeti din cele 2 si de ce?
# As alege filtrul Butterworth pentru ca filtrul Chebyshev pare sa atenueze prea mult semnalul

# Reproiectati filtrele alefand atat un ordin mai mic, cat si unul mai mare. De asemenea, reproiectati filtrul Chebyshev
# cu alte avalori ale rp si observati efectul. Stabiliti valorile optime ale parametrilor incercati pentru a va atinge scopul.
lower_coef_a_butter, lower_coef_b_butter = signal.butter(3, 0.5, 'low')
higher_coef_a_butter, higher_coef_b_butter = signal.butter(11, 0.5, 'low')
lower_filtered_with_butter = signal.filtfilt(lower_coef_a_butter, lower_coef_b_butter, x_count_data)
higher_filtered_with_butter = signal.filtfilt(higher_coef_a_butter, higher_coef_b_butter, x_count_data)
lower_coef_a_cheby, lower_coef_b_cheby = signal.cheby1(3, 3, 0.5, 'low')
higher_coef_a_cheby, higher_coef_b_cheby = signal.cheby1(7, 7, 0.5, 'low')
lower_filtered_with_cheby = signal.filtfilt(lower_coef_a_cheby, lower_coef_b_cheby, x_count_data)
higher_filtered_with_cheby = signal.filtfilt(higher_coef_a_cheby, higher_coef_b_cheby, x_count_data)
plt.figure(1234567)
plt.plot(x_count_data, label='original')
plt.plot(lower_filtered_with_butter, label='ordin 3 butter')
plt.plot(higher_filtered_with_butter, label='ordin 11 butter')
plt.plot(lower_filtered_with_cheby, label='ordin 3, rp 3 cheby')
plt.plot(higher_filtered_with_cheby, label='ordin 7, rp 7 cheby')
plt.legend()
plt.savefig('Tema6/ex4_f.pdf')
