import numpy as np
from matplotlib import pyplot as plt

def sinusoidal(t, A, f, phi):
    return A * np.sin(2 * np.pi * f * t + phi)

# Exercitiul 1
# a)
# Generati o serie de timp aleatoare de dimensiune N = 1000 care sa fie suma a trei componente: trend, sezon si variatii mici.
# Pentru componenta trend folositi o ecuatie de grad 2, pentru sezon folositi doua frecvente iar pentru variatiile mici folosind zgomot alb gaussian.
# Desenati seria de timp si cele trei componente separat.

N = 1000
t = np.arange(N)
trend = 1e-5 * t ** 2 
sezon =  sinusoidal(t, 1, 1 / 100, 0) + sinusoidal(t, 1, 1 / 50, np.pi / 2)
zgomot = np.random.normal(0, 1, N) / 4

serie = trend + sezon + zgomot

fig, ax = plt.subplots(4, figsize=(6, 8))
fig.subplots_adjust(hspace=1)
ax[0].plot(t, serie)
ax[0].set_title("Seria de timp")
ax[1].plot(t, trend)
ax[1].set_title("Trend")
ax[2].plot(t, sezon)
ax[2].set_title("Sezon")
ax[3].plot(t, zgomot)
ax[3].set_title("Zgomot")
fig.savefig("Tema8/ex1_a.pdf")

# b)
# Calculati vectorul de autocorelatie pentru seria de timp generata.
# Verificati daca este o functie in numpy care sa calculeze aceasta cantitate.
# Incercati sa intelegeti de unde vine aceasta cantitate.
# Desenati vectorul de autocorelatie.
# Formula taken from this site: https://otexts.com/fpp3/acf.html
def autocorrelation_manual(y):
    N = len(y)
    y_mean = np.mean(y)
    y = y - y_mean
    r = np.zeros(N)
    for k in range(N):
        for t in range(k + 1, N):
            r[k] += y[t] * y[t - k]
        r[k] /= np.sum(y ** 2)
    return r

def autocorrelation_numpy(y):
    return np.correlate(y, y, mode='full') / np.sum(y ** 2)

r = autocorrelation_numpy(serie)
# take half of the numpy autocorrelation - dont know why
r = r[len(r) // 2:]
r_manual = autocorrelation_manual(serie)

fig, ax = plt.subplots(2, figsize=(6, 8))
fig.subplots_adjust(hspace=0.2)
ax[0].plot(r)
ax[0].set_title("Autocorelatie numpy")
ax[1].plot(r_manual)
ax[1].set_title("Autocorelatie manual")
fig.savefig("Tema8/ex1_b.pdf")