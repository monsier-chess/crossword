from heapq import heappush, heappop
from typing import Generator, Any, DefaultDict
from enum import Enum
from collections import defaultdict
from copy import deepcopy
from random import shuffle

# Входные файлы
matrix_filename = 'test_matrix.txt'
vocabular_filename = 'russian_nouns.txt'

# "Класс" перечисление, виды клеткок кроссворда
Code = Enum('Code', ['EMPTY', 'BLOCK', 'START'])

# Таблица интерпретаций символов, для конфигурации
ParseTable = dict[str, Code]
DEFAULT_PARSE_TABLE = {
    '_': Code.EMPTY, # 0
    '#': Code.BLOCK, # 1
    '*': Code.START  # 2
}

# Типы данных аргументов (используются для аннотаций)

# Матрица - двумерный массив
Matrix = list[list[Any]]

# Матрица, элементы которой это разыне виды клеток кроссворда
CodeMatrix = list[list[Code]]

# Матрица булевых значений
BoolMatrix = list[list[bool]]

# Слово, элемент кроссворда - задаётся парой координат, длиной и направлением (True - горизональное, False - вертикальное)
Word = tuple[tuple[int, int], int, bool] 

# Матрица слов
WordMatrix = list[list[Word]]

# Вложенные словари (хеш-таблицы) пересечений. CrossTable[word1][word2] = *индекс буквы word1, где word1 пересекается с word2*
CrossTable = DefaultDict[Word, dict[Word, int]]
# Словарь - список строк
Vocabular = list[str]

# Суперсловарь
# Хранит отсортированные массивы номеров строк указанной длины, с указанной буквой по указанному номеру
# SuperVocabular[length, index, letter] = [n1, n2, ...]
# Допустимо также использование 'all' вместо последних двух или трех индексов

class SuperVocabular:
    arr: list[list[int]]
    alf: str
    maxlen: int
    
    def __init__(self, maxlen: int, alfabet: str) -> None:
        self.arr = [[] for i in range((((maxlen) * (maxlen+1) // 2 + maxlen)) * len(alfabet))]
        self.maxlen = maxlen
        self.alf = alfabet
    
    def __getitem__(self, args):
        length, index, letter = args
        if length == 'all':
            return self.arr[0]
        if index == 'all':
            return self.arr[(length * (length - 1))//2 * len(self.alf) + 1]
        
        ind1 = (length * (length - 1)) // 2
        ind2 = index + 2
        ind3 = self.alf.index(letter)

        return self.arr[(ind1 + ind2) * len(self.alf) + ind3]

# Алфавит и настраиваемый параметр чувствительности к регистру
RUSSIAN_ALPHABET = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя-'
DEFAULT_CASE_SENSITIVITY = False

# Чтение словаря из файла
def read_vocabular_from_file(filename: str,
                             alfabet: str = RUSSIAN_ALPHABET,
                             case_sensitivity: bool = DEFAULT_CASE_SENSITIVITY) -> Vocabular:
    try:
        vocabular_file = open(filename, 'r')
        
        lines = vocabular_file.readlines()
        if not lines:
            raise ValueError('The file is empty')
        
        if not case_sensitivity:
            alfabet = alfabet.lower()

        vocabular = []
        vocabular_set = set()
        for line_index, line in enumerate(lines, start=1):
            line = line.rstrip()
            if not line:
                continue
            original = line
            if not case_sensitivity:
                line = line.lower()
            for char_index, char in enumerate(line, start=1):
                if char not in alfabet:
                    original_char = original[char_index-1]
                    raise ValueError(f'Invalid character "{original_char}", symbol {char_index} in line {line_index}')
            if line in vocabular_set:
                raise ValueError(f'Word duplication, line {line_index}')
            vocabular_set.add(line)
            vocabular.append(line)
        
        vocabular_file.close()
        if not vocabular_set:
            raise ValueError('File lines contain only whitespace characters')
        
        shuffle(vocabular)
        return vocabular
    
    except FileNotFoundError:
        raise ValueError(f'File not found: {filename}')
    
    except Exception as error:
        raise ValueError(f'Error reading vocabular from file {filename}: {str(error)}')

# Добавление строки в суперсловарь
def add_string_to_super_vocabular(super_vocabular: SuperVocabular, string: str, string_index: int) -> None:
    lenght = len(string)
    super_vocabular['all','all','all'].append(string_index)
    super_vocabular[lenght,'all','all'].append(string_index)
    for letter_index in range(lenght):
        super_vocabular[lenght, letter_index, string[letter_index]].append(string_index)

# Преобразование списка строк в суперсловарь, через последовательное добавление каждого слова
def conver_vocabular_to_super(vocabular: Vocabular, alfabet: str = RUSSIAN_ALPHABET):
    maxlen = max(map(len, vocabular))
    super_vocabular = SuperVocabular(maxlen, alfabet)
    for index, string in enumerate(vocabular):
        add_string_to_super_vocabular(super_vocabular, string, index)
    return super_vocabular

# Чтение матрицы из файла с проверкам корректности символов
def read_matrix_from_file(filename: str,
                          parse_table: ParseTable = DEFAULT_PARSE_TABLE)  -> CodeMatrix:
    try:
        matrix_file = open(filename, 'r')
        
        lines = matrix_file.readlines()
        if not lines:
            raise ValueError('The file is empty')
        
        matrix = []
        line_length = 0
        for line_index, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            if not line_length:
                line_length = len(line)
            if len(line) != line_length:
                raise ValueError('Inconsistent row lengths')
            row = []
            for char_index, char in enumerate(line, start=1):
                if char not in parse_table:
                    raise ValueError(f'Invalid character "{char}", symbol {char_index} in line {line_index}')
                row.append(parse_table[char])
            matrix.append(row)
        
        matrix_file.close()
        return matrix
    
    except FileNotFoundError:
        raise ValueError(f'File not found: {filename}')
    
    except Exception as error:
        raise ValueError(f'Error reading matrix from file {filename}: {str(error)}')

# Проверка на вхождение индексов в границы
def is_ij_in_matrix(matrix: Matrix, i: int, j: int) -> bool:
    return (0 <= i < len(matrix)) and (0 <= j < len(matrix[0]))

# Получение значения, с возвратом по умолчанию при выходе за границы
def soft_get(matrix: Matrix, i: int, j : int, default: Any = None) -> Any:
    if is_ij_in_matrix(matrix, i, j):
        return matrix[i][j]
    return default

# Получение всех соседних клеток. Возрващает генератор
def soft_get_neighbors_coords(matrix: Matrix, i: int, j: int) -> Generator[tuple[int, int], None, None]:
    for delta_i, delta_j in ((1, 0), (0, 1), (-1, 0), (0, 1)):
        y, x = i + delta_i, j + delta_j
        if is_ij_in_matrix(matrix, y, x):
            yield (y, x)

# Получение соседних клеток, перпендикулярных некоторому направлению
def soft_get_perpendicular_neighbors(matrix: Matrix, i: int, j: int, is_horizontal: bool) -> Generator[tuple[int, int], None, None]:
    if is_horizontal:
        delta_mass = [(1, 0)] #, (-1, 0)]
    else:
        delta_mass = [(0, 1)] #, (0, -1)]
    for delta_i, delta_j in delta_mass:
        y, x = i + delta_i, j + delta_j
        if is_ij_in_matrix(matrix, y, x):
            yield (y, x)

# Задать элемент без исключения Out of range
def soft_set(matrix: Matrix, i: int, j : int, value: Any) -> None:
    if is_ij_in_matrix(matrix, i, j):
        matrix[i][j] = value

def is_not_start(matrix: CodeMatrix, i: int, j: int) -> bool:
    return soft_get(matrix, i, j) != Code.START 


def is_not_block(matrix: CodeMatrix, i: int, j: int):
    return soft_get(matrix, i, j, Code.BLOCK) != Code.BLOCK

# Изменение возможных состояний других клеток рядом со словом
def process_word(matrix: CodeMatrix,
                 start_i: int, start_j: int, is_horizontal: bool,
                 ij_can_be_horizonal: BoolMatrix,
                 ij_can_be_vertical: BoolMatrix) -> Word:
    if is_horizontal:
        delta_i, delta_j = 0, 1
        ij_can_be_parallel = ij_can_be_horizonal
        ij_can_be_perpendicular = ij_can_be_vertical
    else:
        delta_i, delta_j = 1, 0
        ij_can_be_parallel = ij_can_be_vertical
        ij_can_be_perpendicular = ij_can_be_horizonal
    
    i, j = start_i, start_j
    lenght = 1
    while (is_not_block(matrix, i, j)):
        ij_can_be_parallel[i][j] = False
        neighbors_coords = soft_get_perpendicular_neighbors(matrix, i, j, is_horizontal)
        for i1, j1 in neighbors_coords:
            ij_can_be_perpendicular[i1][j1] = False
        i += delta_i
        j += delta_j
        lenght += 1
    lenght -= 1
    
    soft_set(ij_can_be_perpendicular, i, j, False)
    soft_set(ij_can_be_parallel, i, j, False)
    
    pre_start_i = start_i - delta_i
    pre_start_j = start_j - delta_j
    soft_set(ij_can_be_perpendicular, pre_start_i, pre_start_j, False)
    soft_set(ij_can_be_parallel, pre_start_i, pre_start_j, False)
    
    word = ((start_i, start_j), lenght, is_horizontal)
    return word

# Подсчёт пересечений в Сross_table
def set_crosses_with_word(word: Word,
                          word_matrix: WordMatrix,
                          cross_table: CrossTable) -> None:
    start_ij, lenght, is_horizontal = word
    start_i, start_j = start_ij
    
    if is_horizontal:
        delta_i, delta_j = 0, 1
    else:
        delta_i, delta_j = 1, 0
    
    i, j = start_i, start_j
    for letter_index in range(lenght):
        if not word_matrix[i][j]:
            word_matrix[i][j] = word
        else:
            other_word = word_matrix[i][j]
            other_i, other_j = other_word[0]
            other_index = abs(i - other_i) or abs(j - other_j)
            cross_table[word][other_word] = letter_index
            cross_table[other_word][word] = other_index
        i += delta_i
        j += delta_j

# Функция обработки считанной матрицы. Учитывает правила и строит CrossTable
# Прим. В перспективе надо добавить проверку на связность кроссворда и на пустые клетки
def process_matrix(matrix: CodeMatrix) -> CrossTable:
    ij_can_be_horizonal = [[True for j in range(len(matrix[i]))] for i in range(len(matrix))]
    ij_can_be_vertical = [[True for j in range(len(matrix[i]))] for i in range(len(matrix))]
    
    word_matrix = [[None for j in range(len(matrix[i]))] for i in range(len(matrix))]
    cross_table = defaultdict(dict)
    word_list = []

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):

            if is_not_start(matrix, i, j):
                continue

            right = ij_can_be_horizonal[i][j] and is_not_block(matrix, i, j+1)
            down = ij_can_be_vertical[i][j] and is_not_block(matrix, i+1, j)
            
            if not right and not down:
                raise ValueError(f'It is impossible to place a word in the cell ({i}, {j}) according to the rules')
            
            if right:
                is_horizontal = True
                word = process_word(matrix, i, j, is_horizontal, ij_can_be_horizonal, ij_can_be_vertical)
                set_crosses_with_word(word, word_matrix, cross_table)
                word_list.append(word)
            
            if down:
                is_horizontal = False
                word = process_word(matrix, i, j, is_horizontal, ij_can_be_horizonal, ij_can_be_vertical)
                set_crosses_with_word(word, word_matrix, cross_table)
                word_list.append(word)
    
    return cross_table, word_list


# Извлечение отсортированных массивов слов из индекса (super_vocabular) по маске без учёта использованных
# Слияние нескольких отсортированных массивов с помощью массива указателей и мин.кучи.
def get_from_super(super_vocabular, mask, used):
    length = len(mask)
    arrays = []
    for index in range(length):
        if mask[index] == '*':
            continue
        array = []
        for elem in super_vocabular[length, index, mask[index]]:
            if used[elem]:
                continue
            array.append(elem)
        if array:
            arrays.append(array)
    
    if not arrays or len(arrays) == 1:
        options = super_vocabular[length, 'all', 'all'] if not arrays else arrays[0]
        result = []
        for num in options:
            if used[num]:
                continue
            result.append(num)
        return result

    heap = []
    for array_index, array in enumerate(arrays):
        if not array:
            return []
        for i in range(len(array)):
            heappush(heap, (array[i], array_index))

    result = []
    counters = [0] * len(arrays)    

    while heap:
        value = arrays[0][counters[0]]
        if not used[value]:    
            for array_index, counter in enumerate(counters):
                if arrays[array_index][counter] != value:
                    break
            else:
                result.append(value)
        
        minval, array_index = heappop(heap)
        counters[array_index] += 1
        if counters[array_index] >= len(arrays[array_index]):
            return result

        while heap and heap[0][0] == minval:
            minval, array_index = heappop(heap)
            counters[array_index] += 1
            if counters[array_index] >= len(arrays[array_index]):
                return result

# Функция решения задачи. Подход рекурсивный.
# На каждом шаге выбирается слово для перебора. По его маске из индекса (суперсловаря) извлекаются подходящие варианты
# Для каждой найденной опции меняются маски соседних слов. Запускается рекурсивный перебор следующего слова.
# Если достигнуто последнее слово и оно решено, то возвращается результат
# Иначе слово помечается как не использованное, текущий результат обновляется, маски заменяются обратно, следующая опция
# В случае неудачи возвращается пустой словарь

def solve(vocalubar: Vocabular,
          super_vocabular: SuperVocabular,
          cross_table: CrossTable,
          mask_table: dict[Word, list[str]],
          word_list: list[Word],
          used: list[bool],
          result: dict[Word, int] = {},
          step: int = 0) -> dict[Word, int]:
    
    if step == len(word_list):
        return result

    word = word_list[step]
    mask = mask_table[word]
    options = get_from_super(super_vocabular, mask, used)
    for option in options:
        result[word] = option
        used[option] = True
        bad_option = False
        string = vocalubar[option]
        if word[1] != len(string):
            bad_option = True
        if not bad_option: 
            for other_word, letter_index in cross_table[word].items():
                other_letter_index = cross_table[other_word][word]
                if result[other_word]:
                    if vocabular[result[other_word]][other_letter_index] != string[letter_index]:
                        bad_option = True
                        break
                else:
                    mask_table[other_word][other_letter_index] = string[letter_index]
        res = []
        if not bad_option:
            res = solve(vocabular, super_vocabular, cross_table, mask_table, word_list, used, result, step+1)
        if res:
            return res
        for other_word, letter_index in cross_table[word].items():
            if result[other_word]:
                continue 
            other_letter_index = cross_table[other_word][word]
            mask_table[other_word][other_letter_index] = '*'
        used[option] = False
        result[word] = None
    return {}


# Подготовка данных для рекурсии и запуск рекурсивного решения
def run(matrix: Matrix, vocabular: Vocabular) -> None:
    cross_table, word_list = process_matrix(matrix)
    super_vocabular = conver_vocabular_to_super(vocabular)

    mask_table = {}
    result = {}
    for word in word_list:
        length = word[1]
        mask_table[word] = ['*'] * length
        result[word] = None

    used = [False] * len(vocabular)
    word_list.sort(key= lambda word: (len(cross_table[word]), word[1]), reverse=True)
    solution = solve(vocabular, super_vocabular, cross_table, mask_table, word_list, used, result)
    
    if len(solution) < len(word_list):
        print('Для указанных входных данных не найдено ни одного полного решения')
        return
    
    print('Использованы слова:\n')
    for i, word in enumerate(solution, start=1):
        option = solution[word]
        print(f' {i}. {vocabular[option]}')

    print('\nЗаполненный кроссворд:\n')
    visual = [['.' for j in range(len(matrix[i]))] for i in range(len(matrix))]
    for word, string_index in solution.items():
        ij, lenght, is_horizontal = word
        i, j = ij
        stirng = vocabular[string_index]
        if is_horizontal:
            for l in range(lenght):
                visual[i][j + l] = stirng[l]
        else:
            for l in range(lenght):
                visual[i+l][j] = stirng[l]
    
    for line in visual:
        print(' ', *line)
    print()

# Чтение данных
matrix = read_matrix_from_file(matrix_filename)
vocabular = read_vocabular_from_file(vocabular_filename)

# Запуск
run(matrix, vocabular)
