import numpy as np
import matplotlib.pyplot as plt
'''
Ex. 1
Declarati o distributie Gaussiana unidimensionala cu media si varianta
data. Esantionati distributia respectiva si afisati intr-un grafic. Repetati
exercitiul dar cu o distributie Gaussiana bidimensionala. Folositi matricea
de covarianta data pe exemplu pe wikipedia si verificati ca aveti acelasi
rezultat. Afisati pe un grafic rezultatul. Pentru a esantiona din distributia
bidimensionala folositi informatia din cursul 10, slide 11.
'''
mean = 0
variance = 1
nrOfSamples = 10000

# Unidimensional
samples = np.random.normal(mean, variance, nrOfSamples)
plt.hist(samples, bins=100)
plt.savefig('Tema10/unidimensional.png')
plt.show()

# Bidimensional
covarianceMatrix = np.array([[1, 3 / 5], [3 / 5, 2]])
mean = np.array([0, 0])
bidimensionalSamples = np.random.multivariate_normal(mean, covarianceMatrix, nrOfSamples)
plt.scatter(bidimensionalSamples[:, 0], bidimensionalSamples[:, 1])
plt.savefig('Tema10/bidimensional.png')
plt.show()
