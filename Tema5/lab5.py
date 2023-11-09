import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import csv

data = np.genfromtxt('Train.csv', delimiter=',', skip_header=1, dtype=None, encoding=None)

# a)
# Frecventa de esantionare este de 1 / 1 ora = 1 / 3600 secunde = 0.0002777777777777778 Hz
row1 = data[0]
row2 = data[1]
date_time1 = datetime.strptime(row1[1], '%d-%m-%Y %H:%M')
date_time2 = datetime.strptime(row2[1], '%d-%m-%Y %H:%M')
delta = date_time2 - date_time1
# fs = 1 / ts
print(str(1 / delta.seconds) + " Hz")

# b)
last_row = data[-1]
date_time_last = datetime.strptime(last_row[1], '%d-%m-%Y %H:%M')

time_interval = date_time_last - date_time1

print(f"Intervalul de timp: {time_interval.days} zile")

# c) ???? fs = 1 / 3600; fs >= 2 * fmax => fmax = fs / 2 = 1 / 7200 Hz

# d)
# put all the values from the count column in a numpy array
count = np.array([row[2] for row in data])

X = np.fft.fft(count)
X = abs(X / len(count))
X = X[:len(count) // 2]

plt.plot(X)
plt.show()



