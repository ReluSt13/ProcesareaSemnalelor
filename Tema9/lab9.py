import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

def sinusoidal(t, A, f, phi):
    return A * np.sin(2 * np.pi * f * t + phi)

# Exercitiul 1
# Importati din laboratorul anterior codul pentru a genera o serie de timp
# aleatoare cu cele tri componente.
np.random.seed(420)

N = 1000
t = np.arange(N)
trend = 1e-5 * t ** 2 
sezon =  sinusoidal(t, 1, 1 / 100, 0) + sinusoidal(t, 1, 1 / 50, np.pi / 2)
zgomot = np.random.normal(0, 1, N) / 4

serie = trend + sezon + zgomot

# Exercitiul 2
# Pentru o serie de timp generata aleator, calculati noua serie rezultata din medierea exponentiala.
# Initial fixati alpha apoi gasiti voi un alpha optim pentru rezultate.

def exponential_smoothing(series, alpha):
    result = [series[0]]
    for t in range(1, len(series)):
        result.append(alpha * series[t] + (1 - alpha) * result[t - 1])
    return result

alpha = 0.5
smoothed = exponential_smoothing(serie, alpha)

fig, ax = plt.subplots(2, figsize=(6, 8))
fig.subplots_adjust(hspace=0.3)
ax[0].plot(t, serie)
ax[0].set_title("Seria de timp")
ax[1].plot(t, smoothed)
ax[1].set_title("Medierea exponentiala")
fig.savefig("Tema9/ex2.pdf")

alphas = np.linspace(0, 1, 100)

best_alpha = None
lowest_mse = float('inf')

for alpha in alphas:
    smoothed = exponential_smoothing(serie[:-1], alpha)
    mse = mean_squared_error(serie[1:], smoothed)
    if mse < lowest_mse:
        best_alpha = alpha
        lowest_mse = mse

print(f'Best alpha: {best_alpha}')

smoothed_with_best_alpha = exponential_smoothing(serie, best_alpha)

fig, ax = plt.subplots(2, figsize=(6, 8))
fig.subplots_adjust(hspace=0.3)
ax[0].plot(t, serie)
ax[0].set_title("Seria de timp")
ax[1].plot(t, smoothed_with_best_alpha)
ax[1].set_title("Medierea exponentiala")
fig.savefig("Tema9/ex2_best_alpha.pdf")

# Exercitiul 3
# Generati un model MA cu orizont q pentru seria de timp utilizata anterior.
# Termenii de eroare epsilon[i] puteti sa ii considerati elemente aleatoare
# extrase din distributia normala standard.
