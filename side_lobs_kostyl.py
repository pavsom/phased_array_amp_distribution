import numba as nb
import numpy as np
# Данная библиотека функций написана костыльным методом, без использования сложных функций и методов. Тем не менее
# поставленные задачи она выполняет, а также показывает достойное быстродействие, в связи с чем активно используется.
# Функции в ней представлены без подробного описания алгоритма их работы.


# Функция поиска первых боковых лепестков ДН, а также их ширины и высоты

# Входные данные:
# F - нормированная двумерная ДН
# az_index0 - индекс максимума ДН по азимуту
# el_index0 - индекс максимума ДН по углу места

# Выходные данные
# az_index_left - индекс первого БЛ слева по азимуту
# az_index_right - индекс первого БЛ справа по азимуту
# el_index_bottom - индекс первого БЛ снизу по углу места
# el_index_top - индекс первого БЛ сверху по углу места
# az_left_width - ширина левого БЛ
# az_right_width - ширина правого БЛ
# el_bottom_height - высота нижнего БЛ
# el_top_height - высота верхнего БЛ


# Декоратор njit, который переводит питон-код в с-код с использованием многопоточности и параллельности
@nb.njit(fastmath=True, nogil=True, cache=True)
def lobs(F, az_index0, el_index0):
    F_az = F[el_index0, :].transpose()
    F_el = F[:, az_index0]

    i = az_index0
    while F_az[i] > F_az[i + 1]:
        i = i + 1
    while F_az[i] < F_az[i + 1]:
        i = i + 1
    az_index_right = i

    j = az_index0
    while F_az[j] > F_az[j - 1]:
        j = j - 1
    while F_az[j] < F_az[j - 1]:
        j = j - 1
    az_index_left = j

    i = el_index0
    while F_el[i] > F_el[i + 1]:
        i = i + 1
    while F_az[i] < F_el[i + 1]:
        i = i + 1
    el_index_bottom = i

    j = el_index0
    while F_el[j] > F_el[j - 1]:
        j = j - 1
    while F_el[j] < F_el[j - 1]:
        j = j - 1
    el_index_top = j

    az_right_width = 0
    i = az_index_right
    while F_az[i] > F_az[i + 1]:
        az_right_width = az_right_width + 1
        i = i + 1
    j = az_index_right
    while F_az[j] > F_az[j - 1]:
        az_right_width = az_right_width + 1
        j = j - 1

    az_left_width = 0
    i = az_index_left
    while F_az[i] > F_az[i + 1]:
        az_left_width = az_left_width + 1
        i = i + 1
    j = az_index_left
    while F_az[j] > F_az[j - 1]:
        az_left_width = az_left_width + 1
        j = j - 1

    el_bottom_height = 0
    i = el_index_bottom
    while F_el[i] > F_el[i + 1]:
        el_bottom_height = el_bottom_height + 1
        i = i + 1
    j = el_index_bottom
    while F_el[j] > F_el[j - 1]:
        el_bottom_height = el_bottom_height + 1
        j = j - 1

    el_top_height = 0
    i = el_index_top
    while F_el[i] > F_el[i + 1]:
        el_top_height = el_top_height + 1
        i = i + 1
    j = el_index_top
    while F_el[j] > F_el[j - 1]:
        el_top_height = el_top_height + 1
        j = j - 1

    if az_left_width % 2 != 0:
        az_left_width = az_left_width - 1

    if az_right_width % 2 != 0:
        az_right_width = az_right_width - 1

    if el_bottom_height % 2 != 0:
        el_bottom_height = el_bottom_height - 1

    if el_top_height % 2 != 0:
        el_top_height = el_top_height - 1

    return [az_index_left, az_index_right, el_index_bottom, el_index_top, az_left_width, az_right_width,
            el_bottom_height, el_top_height]


# Функция поиска уровня первого БЛ

# Входные данные:
# F_az - сечение нормированной ДН в плоскости азимута, дБ
# F_el - сечение нормированной ДН в плоскости угла места, дБ
# az_index - индекс максимума ДН по азимуту
# el_index - индекс максимума ДН по углу места

# Выходные данные:
# F_az[az_index[0]][0] - уровень первого БЛ по азимуту, дБ
# F_el[el_index[0]][0] - уровень первого БЛ по углу места, дБ


# Декоратор njit, который переводит питон-код в с-код
@nb.njit(fastmath=True, nogil=True, cache=True)
def first_lob_level(F_az, F_el, az_index, el_index):

    while F_az[az_index] > F_az[az_index + 1]:
        az_index = az_index + 1
    while F_az[az_index] < F_az[az_index + 1]:
        az_index = az_index + 1

    while F_el[el_index] > F_el[el_index + 1]:
        el_index = el_index + 1
    while F_el[el_index] < F_el[el_index + 1]:
        el_index = el_index + 1

    return F_az[az_index[0]][0], F_el[el_index[0]][0]


# Функция вычисления срднего УБЛ

# Входные данные:
# F - двумерная нормированная ДН, дБ
# az_index - индекс максимума ДН по азимуту
# el_index - индекс максимума ДН по углу места

# Выходные данные:
# avg_sll - средний УБЛ ДН, дБ

def avg_side_lob_level(F, az_index, el_index):
    i = el_index
    j = az_index

    while F[i, j] > F[i, j + 1]:
        j = j + 1

    az_index_right = j
    j = az_index

    while F[i, j] > F[i, j - 1]:
        j = j - 1

    az_index_left = j
    j = az_index

    while F[i, j] > F[i + 1, j]:
        i = i + 1

    el_index_bottom = i
    i = el_index

    while F[i, j] > F[i - 1, j]:
        i = i - 1

    el_index_top = i

    az = np.arange(az_index_left, az_index_right + 1)
    el = np.arange(el_index_top, el_index_bottom + 1)
    counter = 0
    summ = 0.0

    for i in range(np.size(F, 0)):
        for j in range(np.size(F, 1)):
            if (i != el.any()) and (j != az.any()):
                counter = counter + 1
                summ = summ + F[i, j]

    avg_sll = summ / counter

    return avg_sll
