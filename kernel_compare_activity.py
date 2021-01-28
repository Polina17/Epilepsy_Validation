# Скрипт для сравнения валидации биологических и симуляционных данных
# при применении различных типов ядер к фильтрованным нормализованным данным
# при одинаковых сглаживающих параметрах.

# Косинусное ядро при одинаковых h даёт более высокие p_value

import math
import matplotlib.pyplot as plt
import scipy.integrate
from numpy import inf, exp
import scipy.io


# функция для нахождения среднего значения выборки
def mean(array):
    sum = 0
    for elem in array:
        sum += elem
    result = sum/len(array)
    return result


# функция для нахождения среднего квадратического отклонения выборки
def standart_deviation(array):
    sqr_sum = 0
    for elem in array:
        sqr_sum += math.pow(elem, 2)
    result = sqr_sum/len(array) - math.pow(mean(array), 2)
    return math.sqrt(result)


# функция для нормализации данных (минус среднее / стандартное отклонение)
def normal(array):   # нормализуем исходный массив
    result = []
    mean_value = mean(array)
    standart_dev_value = standart_deviation(array)
    for elem in array:
        e = (elem-mean_value)/standart_dev_value
        result.append(e)
    return result


# функция для нахождения суммы значений нормального ядра в данной точке
# array - выборка, x - значение, в котором ищем плотность, h - сглаживающий параметр
def kernel_normal_sum(array, x, h):
    sum = 0
    for elem in array:
        u = (x-elem)/h
        K = math.pow(math.pi * 2, -0.5) * math.pow(math.e, -0.5 * math.pow(u, 2))
        sum += K
    return sum


# функция для нахождения суммы значений ядра Епанечникова в данной точке
# array - выборка, x - значение, в котором ищем плотность, h - сглаживающий параметр
def kernel_Epanechnikov_sum(array, x, h):
    sum = 0
    for elem in array:
        u = (x - elem)/h
        if ((1-math.pow(u, 2)) > 0):
            K = 0.75 * (1-math.pow(u, 2))
        else:
            K = 0
        sum +=K
    return sum


# функция для нахождения суммы значений прямоугольного ядра в данной точке
# array - выборка, x - значение, в котором ищем плотность, h - сглаживающий параметр
def kernel_rectangular_sum(array, x, h):
    sum = 0
    for elem in array:
        u = (x - elem)/h
        if (math.fabs(u) <= 1):
            K = 0.5
        else:
            K = 0
        sum +=K
    return sum


# функция для нахождения суммы значений косинусного ядра в данной точке
# array - выборка, x - значение, в котором ищем плотность, h - сглаживающий параметр
def kernel_cos_sum(array, x, h):
    sum = 0
    for elem in array:
        u = (x - elem)/h
        if (math.fabs(u) <= 1):
            K = (math.pi/4) * math.cos(u*math.pi/2)
        else:
            K = 0
        sum +=K
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
        elif (type == 'Epanechnikov'):
            result.append(kernel_Epanechnikov_sum(array, elem, h) * (1 / (len(array) * h)))
        elif (type == 'rectangular'):
            result.append(kernel_rectangular_sum(array, elem, h) * (1 / (len(array) * h)))
        else:
            result.append(kernel_cos_sum(array, elem, h) * (1 / (len(array) * h)))
    return result


# функция для нахождения значения ядра данного типа в данной точке
def K_value(u, type):
    if (type == 'normal'):
        K = math.pow(math.pi * 2, -0.5) * math.pow(math.e, -0.5 * math.pow(u, 2))
    elif (type == 'Epanechnikov'):
        if ((1 - math.pow(u, 2)) > 0):
            K = 0.75 * (1 - math.pow(u, 2))
        else:
            K = 0
    elif (type == 'rectangular'):
        if (math.fabs(u) <= 1):
            K = 0.5
        else:
            K = 0
    else:
        if (math.fabs(u) <= 1):
            K = (math.pi / 4) * math.cos(u * math.pi / 2)
        else:
            K = 0
    return K


# функция для нахождения внутрипараметрического различия
# биологических данных для нахождения тестовой статистики
def difference_bio(bio, h_bio, type):
    sum = 0
    for elem in bio:
        for el in bio:
            u = (elem-el) / h_bio
            K = K_value(u, type)
            sum += K
    result = sum / (math.pow(len(bio), 2) * h_bio)
    return result


# функция для нахождения внутрипараметрического различия
# симуляционных данных для нахождения тестовой статистики
def difference_neuron(neuron, h_neuron, type):
    sum = 0
    for elem in neuron:
        for el in neuron:
            u = (elem - el) / h_neuron
            K = K_value(u, type)
            sum += K
    result = sum / (math.pow(len(neuron), 2) * h_neuron)
    return result


# функция для нахождения межпараметрического различия симуляционных и биологических данных
# при сглаживающем параметре для био
# для нахождения тестовой статистики
def difference_bio_neuron_h_bio(bio, neuron, h_bio, type):
    sum = 0
    for elem in bio:
        for el in neuron:
            u = (elem - el) / h_bio
            K = K_value(u, type)
            sum += K
    result = sum / (len(bio) * len(neuron) * h_bio)
    return result


# функция для нахождения межпараметрического различия симуляционных и биологических данных
# при сглаживающем параметре для симуляционных данных
# для нахождения тестовой статистики
def difference_bio_neuron_h_neuron(bio, neuron, h_neuron, type):
    sum = 0
    for elem in bio:
        for el in neuron:
            u = (elem - el) / h_neuron
            K = K_value(u, type)
            sum += K
    result = sum / (len(bio) * len(neuron) * h_neuron)
    return result


# функция для нахождения тестовой статистики при заданных сглаживающих параметрах
def statistics(bio, neuron, h_bio, h_neuron, type):
    T = difference_bio(bio, h_bio, type) + difference_neuron(neuron, h_neuron, type) - difference_bio_neuron_h_bio(bio, neuron, h_bio, type) - difference_bio_neuron_h_neuron(bio, neuron, h_neuron, type)
    return T


# функция для нахождения значения p_value при заданных сглаживающих параметрах
def p_value(bio, neuron, h_bio, h_neuron, type):
    T = statistics(bio, neuron, h_bio, h_neuron, type)
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

# Получаем био данные
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

values_array = []

density_estim_bio_normal = []
density_estim_bio_Epanechnikov = []
density_estim_bio_rectangular = []
density_estim_bio_cos = []

density_estim_sim_normal = []
density_estim_sim_Epanechnikov = []
density_estim_sim_rectangular = []
density_estim_sim_cos = []

p_normal = []
p_Epanechnikov = []
p_rectangular = []
p_cos = []

h = [0.1, 0.2]  # h_bio, h_sim

for i in range(0, 15, 7):  # сенсоры 1, 8, 15
    bio_data_sort = sorted(normal(bio_data[i]), key=float)  # сортируем нормализованные данные
    sim_data_sort = sorted(normal(sim_data[i]), key=float)

    print(bio_data_sort)
    print(sim_data_sort)

    bio_test = []  # убираем интервал около нуля, в котором сосредоточено большое кол-во данных
    for elem in bio_data_sort:
        if elem > 0.1 or elem < -0.1:  # [-0.1; 0.1] или [-1; 1] - фильтруем
            bio_test.append(elem)

    sim_test = []
    for elem in sim_data_sort:
        if elem > 0.1 or elem < -0.1:
            sim_test.append(elem)

    min_v = min(min(bio_test), min(sim_test))
    max_v = max(max(bio_test), max(sim_test))

    # print(min_v, max_v)

    values = []  # формируем массив значений, в которых будем искать плотность
    step = (max_v - min_v) / 3000  # число зависит от объёма выборки

    for j in range(3001):
        value = min_v + (step * j)
        values.append(value)  # массив значений для рассчёта плотности

    values_array.append(values)
    print()

    density_estim_bio_normal.append(density_estim(bio_test, h[0], values, 'normal'))
    density_estim_bio_Epanechnikov.append(density_estim(bio_test, h[0], values, 'Epanechnikov'))
    density_estim_bio_rectangular.append(density_estim(bio_test, h[0], values, 'rectangular'))
    density_estim_bio_cos.append(density_estim(bio_test, h[0], values, 'cos'))

    density_estim_sim_normal.append(density_estim(sim_test, h[1], values, 'normal'))
    density_estim_sim_Epanechnikov.append(density_estim(sim_test, h[1], values, 'Epanechnikov'))
    density_estim_sim_rectangular.append(density_estim(sim_test, h[1], values, 'rectangular'))
    density_estim_sim_cos.append(density_estim(sim_test, h[1], values, 'cos'))

    print('h_bio ', h[0], 'h_sim ', h[1])

    p_normal.append(p_value(bio_test, sim_test, h[0], h[1],  'normal'))
    p_Epanechnikov.append(p_value(bio_test, sim_test, h[0], h[1], 'Epanechnikov'))
    p_rectangular.append(p_value(bio_test, sim_test, h[0], h[1], 'rectangular'))
    p_cos.append(p_value(bio_test, sim_test, h[0], h[1], 'cos'))
    print('p-value normal ', p_normal)
    print('p-value Epanechnikov ', p_Epanechnikov)
    print('p-value rectangular ', p_rectangular)
    print('p-value cos ', p_cos)

# Отрисовка графиков
fig, ax = plt.subplots(nrows=6, ncols=2)
plt.suptitle('Симуляция 2 / Эксперимент 21. Запись 30. Сенсоры 1, 8, 15. h_bio = %.2f, h_sim = %.2f' % (h[0], h[1]))
ax[0, 0].set_title('Нормальное ядро')
ax[0, 0].plot(values_array[0], density_estim_bio_normal[0], 'g', label='bio', linewidth=0.8)
ax[0, 0].plot(values_array[0], density_estim_sim_normal[0], 'b', label='sim, p=%.5f' % p_normal[0], linewidth=0.8)
ax[1, 0].plot(values_array[1], density_estim_bio_normal[1], 'g', label='bio', linewidth=0.8)
ax[1, 0].plot(values_array[1], density_estim_sim_normal[1], 'b', label='sim, p=%.5f' % p_normal[1], linewidth=0.8)
ax[2, 0].plot(values_array[2], density_estim_bio_normal[2], 'g', label='bio', linewidth=0.8)
ax[2, 0].plot(values_array[2], density_estim_sim_normal[2], 'b', label='sim, p=%.5f' % p_normal[2], linewidth=0.8)
ax[2, 0].get_xaxis().set_visible(False)
ax[3, 0].set_title('Ядро Епанечникова', pad = 0)
ax[3, 0].plot(values_array[0], density_estim_bio_Epanechnikov[0], 'g', label='bio', linewidth=0.8)
ax[3, 0].plot(values_array[0], density_estim_sim_Epanechnikov[0], 'b', label='sim, p=%.5f' % p_Epanechnikov[0], linewidth=0.8)
ax[4, 0].plot(values_array[1], density_estim_bio_Epanechnikov[1], 'g', label='bio', linewidth=0.8)
ax[4, 0].plot(values_array[1], density_estim_sim_Epanechnikov[1], 'b', label='sim, p=%.5f' % p_Epanechnikov[1], linewidth=0.8)
ax[4, 0].set_ylabel('Значение плотности вероятности')
ax[5, 0].plot(values_array[2], density_estim_bio_Epanechnikov[2], 'g', label='bio', linewidth=0.8)
ax[5, 0].plot(values_array[2], density_estim_sim_Epanechnikov[2], 'b', label='sim, p=%.5f' % p_Epanechnikov[2], linewidth=0.8)
ax[5, 0].set_xlabel('Потенциал локального поля, мВ')
ax[0, 1].set_title('Прямоугольное ядро')
ax[0, 1].plot(values_array[0], density_estim_bio_rectangular[0], 'g', label='bio', linewidth=0.8)
ax[0, 1].plot(values_array[0], density_estim_sim_rectangular[0], 'b', label='sim, p=%.5f' % p_rectangular[0], linewidth=0.8)
ax[1, 1].plot(values_array[1], density_estim_bio_rectangular[1], 'g', label='bio', linewidth=0.8)
ax[1, 1].plot(values_array[1], density_estim_sim_rectangular[1], 'b', label='sim, p=%.5f' % p_rectangular[1], linewidth=0.8)
ax[2, 1].plot(values_array[2], density_estim_bio_rectangular[2], 'g', label='bio', linewidth=0.8)
ax[2, 1].plot(values_array[2], density_estim_sim_rectangular[2], 'b', label='sim, p=%.5f' % p_rectangular[2], linewidth=0.8)
ax[2, 1].get_xaxis().set_visible(False)
ax[3, 1].set_title('Косинусное ядро', pad = 0)
ax[3, 1].plot(values_array[0], density_estim_bio_cos[0], 'g', label='bio', linewidth=0.8)
ax[3, 1].plot(values_array[0], density_estim_sim_cos[0], 'b', label='sim, p=%.5f' % p_cos[0], linewidth=0.8)
ax[4, 1].plot(values_array[1], density_estim_bio_cos[1], 'g', label='bio', linewidth=0.8)
ax[4, 1].plot(values_array[1], density_estim_sim_cos[1], 'b', label='sim, p=%.5f' % p_cos[1], linewidth=0.8)
ax[5, 1].plot(values_array[2], density_estim_bio_cos[2], 'g', label='bio', linewidth=0.8)
ax[5, 1].plot(values_array[2], density_estim_sim_cos[2], 'b', label='sim, p=%.5f' % p_cos[2], linewidth=0.8)

for i in range(6):
    for j in range(2):
        ax[i, j].legend(loc=2, fontsize=8)

#ax[2, 0].legend(loc=2, fontsize=8)
#ax[5, 0].legend(loc=2, fontsize=8)
#ax[2, 1].legend(loc=2, fontsize=8)
#ax[5, 1].legend(loc=2, fontsize=8)
plt.show()
