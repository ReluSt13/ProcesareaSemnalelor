import numpy as np
from matplotlib import pyplot as plt

def sinusoidal(t, A, f, phi):
    return A * np.sin(2 * np.pi * f * t + phi)

# Exercitiul 1
# a)
# Generati o serie de timp aleatoare de dimensiune N = 1000 care sa fie suma a trei componente: trend, sezon si variatii mici.
# Pentru componenta trend folositi o ecuatie de grad 2, pentru sezon folositi doua frecvente iar pentru variatiile mici folosind zgomot alb gaussian.
# Desenati seria de timp si cele trei componente separat.
np.random.seed(42)

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
fig.subplots_adjust(hspace=0.3)
ax[0].plot(r)
ax[0].set_title("Autocorelatie numpy")
ax[1].plot(r_manual)
ax[1].set_title("Autocorelatie manual")
fig.savefig("Tema8/ex1_b.pdf")

# c)
# Calculati un model AR de dimensiune p pentru seria de timp calcilata anterior.
# Afisati pe ecran seria de timp originala si predictiile.

p = 100

train_size = int(0.9 * len(serie))
train_data = serie[:train_size]
test_data = serie[train_size:]

# Create the matrix Y
matrix = np.zeros((len(train_data) - p, p))
for i in range(p):
    matrix[:, i] = train_data[i:-p + i]

# Use np.linalg.lstsq to solve for x in the equation y = Y * x
y = train_data[p:]
x, _, _, _ = np.linalg.lstsq(matrix, y, rcond=None)


# Predict the next len(test_data) values
# After getting one prediction, use it to predict the next value
predictions = np.zeros(len(test_data))
data = train_data.copy().tolist()  # Create a copy of train_data

for i in range(len(test_data)):
    for j in range(p):
        predictions[i] += x[-j - 1] * data[len(data) - j - 1]
    data.append(predictions[i])  # Add the prediction to data

plt.figure(figsize=(6, 4))
plt.plot(train_data, label="Train data")
plt.plot(np.arange(len(train_data), len(serie)), predictions, label="Predictions p = 100")
plt.plot(np.arange(len(train_data), len(serie)), test_data, label="Test data")
plt.legend()
plt.savefig("Tema8/ex1_c.pdf")