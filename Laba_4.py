# Формируется матрица F следующим образом: скопировать в нее А и если в С количество положительных элементов в четных
# столбцах больше, чем количество отрицательных  элементов в нечетных столбцах, то поменять местами С и В симметрично,
# иначе С и Е поменять местами несимметрично. При этом матрица А не меняется. После чего если определитель матрицы А
# больше суммы диагональных элементов матрицы F, то вычисляется выражение: A*AT – K * F*A-1, иначе вычисляется
# выражение (К*A-1 +G-FТ)*K, где G-нижняя треугольная матрица, полученная из А. Выводятся по мере формирования А, F
# и все матричные операции последовательно.
#
#   E    B
#   D    C

import numpy as np
import matplotlib.pyplot as plt

K = int(input('Введите K: '))
N = int(input('Введите N: '))

if N < 2:
    print('Ошибка в исходных данных. Длина сторон матрицы А (N,N) должна быть больше 2!')
    exit()

A = np.random.randint(low=-10, high=11, size=(N, N))

n = N // 2  # размерность матриц B, C, D, E (n x n)

w = N // 2
if N % 2 == 0:
    E = A[0:w, 0:w]
    B = A[0:w, w:]
    C = A[w:, w:]
    D = A[w:, 0:w]
else:
    E = A[0:w, 0:w]
    B = A[0:w, w + 1:]
    C = A[w + 1:, w + 1:]
    D = A[w + 1:, 0:w]

# печатаем матрицы E, B, C, D, A
print('____________________________________________________')
print('Матрица A:')
print(A)
print('____________________________________________________')
print('Матрица E:')
print(E)
print('____________________________________________________')
print('Матрица B:')
print(B)
print('____________________________________________________')
print('Матрица C:')
print(C)
print('____________________________________________________')
print('Матрица D:')
print(D)
print('____________________________________________________')
count_positive_c_even = 0  # количество нулевых элементов в четных столбцах
count_negative_c_odd = 0  # количество нулевых элементов в нечетных столбцах

for j in range(1, n, 2):  # четные столбцы
    for i in range(n):
        if C[i][j] > 0:
            count_positive_c_even += 1

for j in range(0, n, 2):  # нечетные столбцы
    for i in range(n):
        if C[i][j] < 0:
            count_negative_c_odd += 1

F = A.copy()
if count_positive_c_even > count_negative_c_odd:
    print('')
    print('')
    print('Меняем местами B и C симметрично')
    if N % 2 == 0:
        F[0:w, w:] = np.flipud(C)  # flipud - отражение по вертикали,
        F[w:, w:] = np.flipud(B)  # fliplr - по горизонтали, flip - относительно вертикали и горизонтали
    else:
        F[0:w, w + 1:] = np.flipud(C)
        F[w + 1:, w + 1:] = np.flipud(B)
else:
    print('')
    print('')
    print('Меняем местами E и C несимметрично')
    if N % 2 == 0:
        F[0:w, 0:w] = C
        F[w:, w:] = E
    else:
        F[0:w, 0:w] = C
        F[w + 1:, w + 1:] = E
print('')
print('____________________________________________________')
print('Матрица F')
print(F)

det_A = np.linalg.det(A)  # определитель матрицы A
sum_diag = np.trace(F)  # сумма диагональных элементов матрицы F

if det_A > sum_diag:  # определитель матрицы A больше суммы диагональных элементов матрицы F
    print('определитель матрицы A больше суммы диагональных элементов матрицы F')
    result = A.dot(A.T) - K * F.dot(np.linalg.inv(A))
else:
    G = np.tril(A, -1)  # нижняя трегугольная матрица из матрицы A
    result = (K * np.linalg.inv(A) + G - F.T) * K

np.set_printoptions(precision=1, suppress=True)  # выводим с точностью до одного знака после запятой и без e
print('Результат:')
print(result)

# работа с графиками
plt.figure(figsize=(16, 9))

# вывод тепловой карты матрицы F
plt.subplot(2, 2, 1)
plt.xticks(ticks=np.arange(F.shape[1]))
plt.yticks(ticks=np.arange(F.shape[1]))
plt.xlabel('Номер столбца')
plt.ylabel('Номер строки')
hm = plt.imshow(F, cmap='Oranges', interpolation="nearest")
plt.colorbar(hm)
plt.title('Тепловая карта элементов')

# вывод диаграммы распределения сумм элементов по строкам в матрице F
sum_by_rows = np.sum(F, axis=1)  # axis = 1 - сумма по строкам
x = np.arange(F.shape[1])
plt.subplot(2, 2, 2)
plt.plot(x, sum_by_rows, label='Сумма элементов по строкам')
plt.xlabel('Номер строки')
plt.ylabel('Сумма элементов')
plt.title('График суммы элементов по строкам')
plt.legend()

# вывод диаграммы распределения количества положительных элементов в столбцах матрицы F
res = []
for col in F.T:
    count = 0
    for el in col:
        if el > 0:
            count += 1
    res.append(count)

x = np.arange(F.shape[1])
plt.subplot(2, 2, 3)
plt.bar(x, res, label='Количество положительных элементов в столбцах')
plt.xlabel('Номер столбца')
plt.ylabel('Количество положительных элементов')
plt.title('График количества положительных элементов в столбцах')
plt.legend()

# вывод круговой диаграммы
x = np.arange(F.shape[1])
plt.subplot(2, 2, 4)
P = []
for i in range(N):
    P.append(abs(F[0][i]))
plt.pie(P, labels=x, autopct='%1.2f%%')
plt.title("График с использованием функции pie")

plt.tight_layout(pad=3.5, w_pad=3, h_pad=4) # расстояние от границ и между областями
plt.suptitle("Использование библиотеки Matplotlib", y=1)
plt.show()