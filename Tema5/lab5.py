import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import csv

data = np.genfromtxt('Train.csv', delimiter=',', skip_header=1, dtype=None, encoding=None)

dates = np.array([datetime.strptime(row[1], '%d-%m-%Y %H:%M') for row in data])
# a)
# Frecventa de esantionare este de 1 / 1 ora = 1 / 3600 secunde = 0.0002777777777777778 Hz
row1 = data[0]
row2 = data[1]
date_time1 = datetime.strptime(row1[1], '%d-%m-%Y %H:%M')
date_time2 = datetime.strptime(row2[1], '%d-%m-%Y %H:%M')
delta = date_time2 - date_time1
# fs = 1 / ts
print(str(1 / delta.seconds) + " Hz")
fs = 1 / delta.seconds

# b)
last_row = data[-1]
date_time_last = datetime.strptime(last_row[1], '%d-%m-%Y %H:%M')

time_interval = date_time_last - date_time1

print(f"Intervalul de timp: {time_interval.days} zile")

# c) fs = 1 / 3600; fs >= 2 * fmax => fmax = fs / 2 = 1 / 7200 Hz (???)

# d)
# put all the values from the count column in a numpy array
count = np.array([row[2] for row in data])
N = len(count)
X = np.fft.fft(count)
tot_X = X

X = abs(X / N)
X = X[:N // 2]
# f_fftfrecv = np.fft.fftfreq(N, 1 / fs)
# f_fftfrecv = f_fftfrecv[:N // 2]
f = fs * np.linspace(0, N //2, N // 2) / N
plt.plot(f, X)
plt.savefig('d.pdf')

# e)
if tot_X[0] > 0:
    print("Semnalul are o componenta continua")

    copy_X = tot_X
    copy_X[0] = 0

    count_modified = np.fft.ifft(copy_X)

    plt.figure(123)
    plt.plot(count_modified.real)
    plt.savefig('modified_signal.pdf')
else:
    print("Semnalul nu are o componenta continua")

# f)
X[0] = 0
indiciSortati = np.argsort(X)
primele4Frecvente = indiciSortati[-4:] * fs / N

print(primele4Frecvente)

print((1 / primele4Frecvente) / 3600 / 24) # 1 / frecventa = perioada in secunde --> transformam in zile
# a doua perioada pare sa fie de o zi, urmatoarele doua posibil sa fie un an si doi ani... prima nu stiu sigur ce reprezinta

# g)
# 1 iulie 2014 - zi de luni
# print(np.where(dates == datetime.strptime('01-07-2014 00:00', '%d-%m-%Y %H:%M'))[0][0])
start = np.where(dates == datetime.strptime('01-07-2014 00:00', '%d-%m-%Y %H:%M'))[0][0]
end = np.where(dates == datetime.strptime('01-8-2014 00:00', '%d-%m-%Y %H:%M'))[0][0]
plt.figure(1234)
plt.plot(count[start:end])
plt.savefig('g.pdf')

# h) ????

# i)
threshhold_freq = np.sort(f)[-len(f) // 10:][0]
f_copy = f.copy()
f = f[f < threshhold_freq]
X = X[f_copy < threshhold_freq]
plt.figure(12345)
plt.plot(f, X)
plt.savefig('i.pdf')

filtered_signal = np.fft.ifft(X)

plt.figure(12346)
plt.plot(filtered_signal.real)
plt.savefig('filtered_signal.pdf')