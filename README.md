# Laba_4
Формируется матрица F следующим образом: скопировать в нее А и если в С количество положительных элементов в четных
столбцах больше, чем количество отрицательных  элементов в нечетных столбцах, то поменять местами С и В симметрично,
иначе С и Е поменять местами несимметрично. При этом матрица А не меняется. После чего если определитель матрицы А
больше суммы диагональных элементов матрицы F, то вычисляется выражение: A*AT – K * F*A-1, иначе вычисляется
выражение (К*A-1 +G-FТ)*K, где G-нижняя треугольная матрица, полученная из А. Выводятся по мере формирования А, F
и все матричные операции последовательно.

E    B
D    C
