import numpy as np
from matplotlib import pyplot as plt

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
# Ce observati? (COMPLETEAZA AICI)
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
