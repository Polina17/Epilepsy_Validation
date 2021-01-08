# Скрипт для нахождения p_value между био и симуляционными данными
# при применении нормального ядра к исходным данным с построением графиков

import math
import matplotlib.pyplot as plt
import scipy.integrate
from numpy import inf, exp
import scipy.io


# функция для нахождения суммы значений нормального ядра в данной точке
# array - выборка, x - значение, в котором ищем плотность, h - сглаживающий параметр
def kernel_normal_sum(array, x, h):  # array - исх. нормализованная выборка, x - значение, в котором ищем плотность
    sum = 0
    for elem in array:
        u = (x-elem)/h
        K = math.pow(math.pi * 2, -0.5) * math.pow(math.e, -0.5 * math.pow(u, 2))
        sum += K
    return sum


# функция для нахождения значений плотности вероятности в данных точках
# array - выборка
# h - сглаживающий параметр
# values - точки, в которых ищем плотность
# type - тип ядра
def density_estim(array, h, values, type):
    result = []
    for elem in values:
        if (type == 'normal'):
            result.append(kernel_normal_sum(array, elem, h) * (1/(len(array)*h)))
    return result


# функция для нахождения внутрипараметрического различия
# биологических данных для нахождения тестовой статистики
def difference_bio(bio, h_bio):
    sum = 0
    for elem in bio:
        for el in bio:
            u = (elem-el) / h_bio
            K = math.pow(math.pi * 2, -0.5) * math.pow(math.e, -0.5 * math.pow(u, 2))
            sum += K
    result = sum / (math.pow(len(bio), 2) * h_bio)
    return result


# функция для нахождения внутрипараметрического различия
# симуляционных данных для нахождения тестовой статистики
def difference_neuron(neuron, h_neuron):
    sum = 0
    for elem in neuron:
        for el in neuron:
            u = (elem - el) / h_neuron
            K = math.pow(math.pi * 2, -0.5) * math.pow(math.e, -0.5 * math.pow(u, 2))
            sum += K
    result = sum / (math.pow(len(neuron), 2) * h_neuron)
    return result


# функция для нахождения межпараметрического различия симуляционных и биологических данных
# при сглаживающем параметре для био
# для нахождения тестовой статистики
def difference_bio_neuron_h_bio(bio, neuron, h_bio):
    sum = 0
    for elem in bio:
        for el in neuron:
            u = (elem - el) / h_bio
            K = math.pow(math.pi * 2, -0.5) * math.pow(math.e, -0.5 * math.pow(u, 2))
            sum += K
    result = sum / (len(bio) * len(neuron) * h_bio)
    return result


# функция для нахождения межпараметрического различия симуляционных и биологических данных
# при сглаживающем параметре для симуляционных данных
# для нахождения тестовой статистики
def difference_bio_neuron_h_neuron(bio, neuron, h_neuron):
    sum = 0
    for elem in bio:
        for el in neuron:
            u = (elem - el) / h_neuron
            K = math.pow(math.pi * 2, -0.5) * math.pow(math.e, -0.5 * math.pow(u, 2))
            sum += K
    result = sum / (len(bio) * len(neuron) * h_neuron)
    return result


# функция для нахождения тестовой статистики при заданных сглаживающих параметрах
def statistics(bio, neuron, h_bio, h_neuron):
    T = difference_bio(bio, h_bio) + difference_neuron(neuron, h_neuron) - difference_bio_neuron_h_bio(bio, neuron, h_bio) - difference_bio_neuron_h_neuron(bio, neuron, h_neuron)
    return T


# функция для нахождения значения p_value при заданных сглаживающих параметрах
def p_value(bio, neuron, h_bio, h_neuron):
    T = statistics(bio, neuron, h_bio, h_neuron)
    f = lambda x: exp((-x ** 2)/2)
    integral = scipy.integrate.quad(f, -inf, T)
    result = 1/(math.sqrt(2*math.pi)) * integral[0]
    return result

# Получаем симуляционные данные
sim_data = []
f = open('/home/polina/диплом/эпилепсия_данные_sim/sim_data_prepare2', 'r')

elem = []
for line in f:
    elem.append(float(line))

# берём каждое 10-е значение, т.к. био данные - по мс, сим - по 0.1 мс
for i in range(15):  # 0-600, 601-1201
    sim_data.append(elem[i*601+1 : (i*601)+601 : 10])  # получили sim data по сенсорам
print('длина sim ', len(sim_data))

# получаем данные био
mat = scipy.io.loadmat('/home/polina/диплом/эпилепсия_данные_био/2011 may 03 P32 BCX rust/2011_05_03_0021.mat', squeeze_me=True)
data = mat['lfp']
print(data.shape)

bio_data = []
for j in range(15):
    result = []
    for i in range(2000):
        result.append(data[i, j, 30])  # № записи
    bio_data.append(result)

print('длина bio ', len(bio_data))  # получили lfp био по сенсорам - 15 массивов по 2000 данных

print()

fig = plt.figure(1) #первое окно с графиками

h = [[20, 80], [20, 80], [20, 80], [20, 80], [20, 80]]  # h_bio, h_sim для каждого сенсора
for i in range(0, 5):  # 0,5  5,10  10,15
    bio_data[i].sort()
    sim_data[i].sort()
#   print(bio_data[i])
 #  print(len(bio_data[i]))
  # print(sim_data[i])
  # print(len(sim_data[i]))

    min_v = min(min(bio_data[i]), min(sim_data[i]))
    max_v = max(max(bio_data[i]), max(sim_data[i]))

    print(min_v, max_v)

    values = []  # формируем массив значений, в которых будем искать плотность
    step = (max_v - min_v) / 3000  # число зависит от объёма выборки

    for j in range(3001):
        value = min_v + (step * j)
        values.append(value)  # массив значений для рассчёта плотности bio

    print()

    density_estim_bio = density_estim(bio_data[i], h[i][0], values, 'normal')   # h[i-5][0]
    density_estim_neuron = density_estim(sim_data[i], h[i][1], values, 'normal')   # h[i-5][1]

    T = statistics(bio_data[i], sim_data[i], h[i][0], h[i][1])
    print('h_bio ', h[i][0], 'h_sim ', h[i][1])
    print('значение статистики ', T)

    p = p_value(bio_data[i], sim_data[i], h[i][0], h[i][1])
    print('p-value ', p)
    print()

    # Отрисока графиков для 5 сенсоров
    plt.subplot(5, 1, i+1)  # i-5+1 i-10+1
    plt.plot(values, density_estim_bio, 'slategrey', label='bio. h_bio = %.1f, h_sim = %.1f' % (h[i][0], h[i][1]))
    plt.plot(values, density_estim_neuron, 'skyblue', label='sim, p-value=%.10f' % p)
    if i == 0:
        plt.title('Симуляция 2 / Эксперимент 21. Запись 30. Сенсоры 1-5')
    if i == 2 :
        plt.ylabel('Значение плотности вероятности')
    if i == 4:
        plt.xlabel('Потенциал локального поля, мВ')
    plt.legend(loc=2)

plt.show()

