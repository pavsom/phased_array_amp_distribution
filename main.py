import function_library as func
import side_lobs_kostyl as kostyl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Блок настраиваемых параметров
f = 1332.5 * 10**6                  # Рабочая частота
c = 299704000                       # Скорость света
az0 = np.radians(0)                 # Направление фазирования по азимуту, градусы
el0 = np.radians(0)                 # Направление фазирования по углу места, градусы
step_degree = 0.05                  # Шаг моделирования ДН, градусы
degree_range = 20                   # Диапазон моделирования ДН, градусы
tilt_angle = np.radians(10)         # Угол подъёма антенной решётки
SLR_X_dB = 20                     # Желаемый УБЛ по азимуту, дБ
SLR_Y_dB = 20                     # Желаемый УБЛ по углу места, дБ
n_param = 5                         # n-параметр, для n-параметрического распределения Тейлора
descrete = 4                        # Линейный размер одного блока излучателей (по умолчанию 4х4)
type = 'dolf_chebyshev'             # Тип желаемого распределения:
                                    # dolf_chebyshev - распаределение Дольф-Чебышева,
                                    # taylor_1_param - однопараметрическое распределение Тейлора
                                    # taylor_n_param - n-параметрическое распределение Тейлора

# Пересчёт частоты в длину волны и волновое число
lamda = c / f                       # Длина волны, м
k = (2 * np.pi) / lamda             # Волновое число, рад/м

# Создание массивов азимута и угла места в радианах, исходя из заданных параметров для расчёта ДН
dots = int(degree_range / step_degree)
el = np.radians(np.linspace(0, degree_range, num=dots, endpoint=True))
az = np.radians(np.linspace(-degree_range / 2, degree_range / 2, num=dots, endpoint=True))

# Пересчё азимута и угла места в UV-координаты для расчёт ДН
u = np.multiply(np.sin(az), np.cos(el))
v = np.sin(el) * np.cos(tilt_angle) - np.sin(tilt_angle) * np.multiply(np.cos(az), np.cos(el))
np.multiply(np.sin(az), np.cos(el))
[U, V] = np.meshgrid(u, v)

# Пересчёт направления фазирования в UV-координаты
u0 = np.multiply(np.sin(az0), np.cos(el0))
v0 = np.sin(el0)

# Чтение координат и номеров* антенных элементов из файла
# * - номера антенных элементов задаются от 1 до максимального количества излучателей, т.к. после чтения происходит
# построчная сортировка излучателей. Фиксация номеров необходимо для обратной сортировки при создании файла амплитудного
# аспределения
[x, y, DATA_X, DATA_Y] = func.data_read('antenna_array.csv')

# Чтение количества излучателей в одном столбце и одной строке
y1 = np.int64(np.linspace(0, len(y) / 2 - 1, num=int(len(y) / 2), endpoint=True))
N1 = len(x)
N2 = len(y1)

# Генерация требуемого амплитудного распределения по полотну антенной решётки
if type == 'taylor_n_param':
    [amp_x, amp_y, A_xy] = func.taylor_n_param(SLR_X_dB, SLR_Y_dB, N1, N2, n_param)
elif type == 'taylor_1_param':
    [amp_x, amp_y, A_xy] = func.taylor_1_param(SLR_X_dB, SLR_Y_dB, N1, N2)
elif type == 'dolf_chebyshev':
    [amp_x, amp_y, A_xy] = func.dolf_chebyshev_distribution(SLR_X_dB, SLR_Y_dB, N1, N2)

# Поскольку блок излучателей представлет собой ячейку из 4х4 (16) излучателей для дискретизации было принято решение
# вычислять среднее арифметическое амплитуд , попадающих в один блок и присваивать всем им полученное значение

# Создание буффера накопления значений амплитуды для вычисления среднего арифметического этих значений
len_x = int(N1 / descrete)
len_y = int((N2 * 2) / descrete)
buff_descr_amp = np.array([[0.0] * len_x] * len_y)

# Накопление значений амплитуды для излучателей из одного блока
for i in range(len_y):
    for j in range(len_x):
        for m in range(descrete):
            for n in range(descrete):
                buff_descr_amp[i, j] += abs(A_xy[m + descrete * i, n + descrete * j])

# Деление накопленных значений на количество излучателей в одном блоке
buff_descr_amp = buff_descr_amp / (descrete ** 2)

# Присваивание всем излучателям каждого блока срдних арифметических значений амплитуды возбуждающих токов всех
# излучателей блока
for i in range(len_y):
    for j in range(len_x):
        for m in range(descrete):
            for n in range(descrete):
                A_xy[m + descrete * i, n + descrete * j] = buff_descr_amp[i, j]

# Задание ДН одного излучателя
self_pattern = np.sqrt(np.sqrt(1 - np.power(u, 2) - np.power(v, 2)))
# Задание равномернорго амплитудного распределения для сравнительного анализа
amp_xy_base = np.array([1.0] * N1 * (2 * N2))
# Вычисление и нормирование исходной ДН с равномерным распределением
F_azel_base = func.DN_form(U, V, amp_xy_base, DATA_X, DATA_Y, k, self_pattern)
F_azel_base_norm = 20 * np.log10(abs(F_azel_base) / abs(F_azel_base).max())
# Вычисление индексов азимута и угла места, соответствующих направлению фазирования
[el_index, az_index] = np.where(F_azel_base_norm == F_azel_base_norm.max())
# Создание сечения ДН в плоскостях азимута и угла места
F_az_base_norm = F_azel_base_norm[el_index, :].transpose()
F_el_base_norm = F_azel_base_norm[:, az_index]

# Вычисление и нормирование новой ДН по синтезировнному амплитудному распределению
F_azel = func.DN_form(U, V, A_xy.flatten(), DATA_X, DATA_Y, k, self_pattern)
F_azel_norm = 20 * np.log10(abs(F_azel) / abs(F_azel).max())
# Создание сечения ДН в плоскостях азимута и угла места
F_az_norm = F_azel_norm[el_index, :].transpose()
F_el_norm = F_azel_norm[:, az_index]

# Расчёт исходного среднего уровня боковых лепестков
# КОСТЫЛЬ
avg_sll = kostyl.avg_side_lob_level(F_azel_norm, az_index, el_index)
# Расчёт нового среднего уровня боковых лепестков
# КОСТЫЛЬ
avg_sll_base = kostyl.avg_side_lob_level(F_azel_base_norm, az_index, el_index)

# Расчёт исходной ширины главного лепестка по азимуту и углу места
width_az, width_el = func.find_delta_3dB(F_azel_norm, az, el)
# Расчёт новой ширины главного лепестка по азимуту и углу места
width_az_base, width_el_base = func.find_delta_3dB(F_azel_base_norm, az, el)

# Расчёт исхондого уровня первого бокового лепестка
# КОСТЫЛЬ
first_lob_az, first_lob_el = kostyl.first_lob_level(F_azel_norm[el_index, :].transpose(), F_azel_norm[:, az_index],
                                                    az_index, el_index)
# Расчёт нового уровня первого бокового лепестка
# КОСТЫЛЬ
first_lob_az_base, first_lob_el_base = kostyl.first_lob_level(F_azel_base_norm[el_index, :].transpose(),
                                                            F_azel_base_norm[:, az_index], az_index, el_index)

# Расчёт потерь в луче в направлении фазирования
P_reduction = 20 * np.log10(abs(F_azel.max()) / abs(F_azel_base).max())

# Блок отображения параметров в консоль
print('Начальная ширина ГЛ по азимуту', round(width_az_base, 2), 'градусов', 'Начальная ширина ГЛ по углу места',
      round(width_el_base, 2), 'градусов')
print('Новая ширина ГЛ по азимуту', round(width_az, 2), 'градусов', 'Новая ширина ГЛ по углу места',
      round(width_el, 2), 'градусов')
print('Расширение главного лепестка по азимуту', round((width_az/width_az_base - 1) * 100, 2), '%')
print('Расширение главного лепестка по углу места', round((width_el/width_el_base - 1) * 100, 2), '%')
print('Начальный уровень первого БЛ в плоскости азимута', round(first_lob_az_base, 2), 'Дб')
print('Начальный уровень первого БЛ в плоскости угла места', round(first_lob_el_base, 2), 'Дб')
print('Новый уровень первого БЛ в плоскости азимута', round(first_lob_az, 2), 'Подавление',
      round(first_lob_az - first_lob_az_base, 2), 'Дб')
print('Новый уровень первого БЛ в плоскости угла места', round(first_lob_el, 2), 'Подавление',
      round(first_lob_el - first_lob_el_base, 2), 'Дб')
print('Начальный средний уровень УБЛ', round(avg_sll_base, 2), 'Дб')
print('Новый средний уровень УБЛ', round(avg_sll, 2), 'Дб')
print('Изменение мощности в направлении максимума ДН', round(P_reduction, 2), 'дБ',
      round(abs(F_azel_base.max()) / abs(F_azel.max()), 2), 'Раз')

# Блок генерации файла амплитудного распределения. Раскоментировать по необходимости
# num_amp_array = np.array([DATA_NUM, A_xy.flatten()])
# columns = ['num', 'amp']
# index = np.arange(1, len(A_xy.flatten()) + 1, dtype=int)
# num_amp_df = pd.DataFrame(num_amp_array.transpose(), index, columns)
# num_amp_df = num_amp_df.sort_values('num')
#
# if type == 'taylor_n_param':
#     name = type + '_' + str(n_param) + '_' + str(SLR_X_dB) + '_dB.txt'
# else:
#     name = type + '_' + str(SLR_X_dB) + '_dB.txt'
#
# amp_sorted = np.array(num_amp_df['amp'])
# np.savetxt(name, amp_sorted, fmt='%f')

# Создание осей с номерами излучателей по вертикали и горизонтали
p_double1 = np.linspace(1, N1, num=N1, endpoint=True)
p_double2 = np.linspace(1, N2, num=N2, endpoint=True)

# Блок потображения графиков
plt.figure(1)
plot_amp_distribution = plt.plot(p_double1, amp_x)
plt.xlim([1, N1])
plt.ylim([0, 1])
plt.xlabel(f'Номер элемента')
plt.ylabel(f'Относительная амплитуда')
plt.title(f'Амплитудное распределение по оси Х')
plt.grid(True)

plt.figure(2)
plot_amp_distribution = plt.plot(p_double2, amp_y)
plt.xlim([1, N2])
plt.ylim([0, 1])
plt.xlabel(f'Номер элемента')
plt.ylabel(f'Относительная амплитуда')
plt.title(f'Амплитудное распределение по оси Y')
plt.grid(True)

plt.figure(3)
plot_F_az = plt.plot(np.degrees(az), F_az_base_norm, np.degrees(az), F_az_norm)
plt.xlim([-10, 10])
plt.ylim([-50, 0])
plt.xlabel(f'Азимут, градусы')
plt.ylabel(f'ДН, дБ')
plt.title(f'ДН в плоскости азимута')
plt.grid(True)
plt.legend(['Исходная ДН', 'Синтезированная ДН', 'Сечение двумерной ДН'])

plt.figure(4)
plot_F_az = plt.plot(np.degrees(el), F_el_base_norm, np.degrees(el), F_el_norm)
plt.xlim([0, 20])
plt.ylim([-50, 0])
plt.xlabel(f'Угол места, градусы')
plt.ylabel(f'ДН, дБ')
plt.title(f'ДН в плоскости угла места')
plt.grid(True)
plt.legend(['Исходная ДН', 'Синтезированная ДН', 'Сечение двумерной ДН'])

plt.figure(5)
levels = np.linspace(-40, 0, 400, endpoint=True)
contours = plt.contour(np.degrees(az), np.degrees(el), F_azel_base_norm, levels=levels)
plt.xlim([-10, 10])
plt.ylim([0, 20])
plt.xlabel(f'Азимут, градусы')
plt.ylabel(f'Угол места, градусы')
plt.grid(True)
cbar = plt.colorbar(contours, label=f'ДН, дБ', drawedges=False)
plt.title(f'Нормированная контурная ДН')

plt.figure(6)
levels1 = np.linspace(-40, 0, 400, endpoint=True)
contours1 = plt.contour(np.degrees(az), np.degrees(el), F_azel_norm, levels=levels1)
plt.xlim([-10, 10])
plt.ylim([0, 20])
plt.xlabel(f'Азимут, градусы')
plt.ylabel(f'Угол места, градусы')
plt.grid(True)
cbar1 = plt.colorbar(contours1, label=f'ДН, дБ', drawedges=False)
plt.title(f'Синтезированная контурная ДН')

plt.show()
