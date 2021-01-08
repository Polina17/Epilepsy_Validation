# Скрипт для вычисления p_value между био- и симуляционными данными
# с использованием критерия Смирнова

import scipy.io
import math


# функция для нахождения накопленной частоты варианты value в выборке array
def cumulative_frequency(value, array):
    if value in array:
        array.sort()
        result = array.index(value) / len(array)
    else:
        array.append(value)
        array.sort()
        result = array.index(value) / (len(array) - 1)
        array.remove(value)
    return result


# Нахождение эмпирических функций распределения выборок
# для каждой варианты в выборках её накопленная частота в 1-й выборке и 2-й
def empirical_function(bio, neuron):
    function = dict()
    for i in bio:
        function[i] = [cumulative_frequency(i, bio), cumulative_frequency(i, neuron)]
    for j in neuron:
        if j not in bio:
            function[j] = [cumulative_frequency(j, bio), cumulative_frequency(j, neuron)]
    return function


# функция для вычисления значения тестовой статистики критерия Смирнова
def statistics_value(bio, neuron):
    func = empirical_function(bio, neuron)
    difference = []
    for i in func:
        difference.append(abs(func[i][0] - func[i][1]))
    D = max(difference)
    n = len(bio)
    m = len(neuron)
    result = D * math.sqrt((n*m)/(n+m))
    return result


# функция для вычисления значения p_value критерия Смирнова
def p_val(statistic):
    Kolm_func = 1 - 2 * (math.e ** (-2 * (statistic ** 2)))
    p = 1 - Kolm_func
    return p

# Получаем симуляционне данные
sim_data = []
f = open('/home/polina/диплом/эпилепсия_данные_sim/sim_data_prepare2', 'r')

elem = []
for line in f:
    elem.append(float(line))

# берём каждое 10-е значение, т.к. био данные - по мс, сим - по 0.1 мс
for i in range(15):  # 0-600, 601-1201
    sim_data.append(elem[i*601+1 : (i*601)+601 : 10])  # получили sim data по сенсорам

sim_data_full = []  # обобщённые симуляционные данные
for elem in sim_data:
    for el in elem:
        sim_data_full.append(el)

# Получаем данные био
mat = scipy.io.loadmat('/home/polina/диплом/эпилепсия_данные_био/2011 may 03 P32 BCX rust/2011_05_03_0023.mat', squeeze_me=True)
data = mat['lfp']
print(data.shape)

for rec in range(15, 20):  # номер записи
    bio_data = []
    for j in range(15):
        result = []
        for i in range(2000):
            result.append(data[i,j,rec])
        bio_data.append(result)

    print('длина bio ', len(bio_data))   # получили lfp био по сенсорам - 15 массивов по 2000 данных

    bio_data_full = []
    for elem in bio_data:
        for el in elem:
            bio_data_full.append(el)

    print()

    print('запись ', rec)

    for i in range(15):
#    print('Сенсор ', i+1)
        statistics = statistics_value(bio_data[i], sim_data[i])
        print(statistics)
        p = 2 * (math.e ** (-2 * (statistics ** 2)))
        print(p)

    print()
    print('Обобщённые данные:')
    statistics = statistics_value(bio_data_full, sim_data_full)
    print(statistics)
    p = 2 * (math.e ** (-2 * (statistics ** 2)))
    print(p)
    print()
