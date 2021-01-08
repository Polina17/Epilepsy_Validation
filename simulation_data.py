# Скрипт для чтения исходных файлов симуляционных данных

import csv


File = open('/home/polina/диплом/эпилепсия_данные/extr_all.csv', newline='')
reader = csv.reader(File)
count = 0
for row in reader:
    count += 1
    print(row)

print()
print(count)