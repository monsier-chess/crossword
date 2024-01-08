# Crossword Solver on Python

В этом репозитории представлен код для работы на альтернативный экзамен "Комбинаторный поиск и эвристика. Составление кроссвордов".
Программа по заданной форме кроссворда и массиву существительных (есть скаченный стандартный) составляет кроссворд, т.е. заполняет ячейки по всем правилам.
Используется переборный рекурсивный алгоритм с некоторыми оптимизациями.

## Формат ввода

### Форма кроссворда
"Форма" кроссорда задаётся в виде матрицы, построчно без пробелов и других разделителей. Длина строк должна быть одинаковой.
Могут использоваться только символы из таблицы декодировки - '#', '_' и '*'.

```python
# Таблица интерпретаций символов, для конфигурации
ParseTable = dict[str, Code]
DEFAULT_PARSE_TABLE = {
    '_': Code.EMPTY, # 0
    '#': Code.BLOCK, # 1
    '*': Code.START  # 2
}
```

'#' это блок, '_' - это продолжение слова, ячейка для буквы, а '*' - старт слова.
Слово после старта может продолжаться только влево или вниз. Вертикальные слова не могут стыковаться или пересекаться вертикально, горизонтальные не могут стыковаться или пересекаться горизонтально.
К началу горизонтального слова нельзя пристыковывать ничего слева (иначе визуально начало слова будет читаться именно от пристыкованной буквы), для вертикальных - сверху.
Если после клетки начала нет возможности провести слово ни вниз, ни влево, то выбрасывается исключение.
Параллельные слова могут касаться друг друга, если соблюдены все остальные правила.

### Словарь
Словарь задаётся как список слов в одном файле, каждое с новой строки.
Не допускается написание каких-либо дополнительных разделителей или символов. Регистр игнорируется, это можно изменить при вызове функции в коде.

### Исключения

Код может выбрасывать некоторые исключения, это ожидаемое поведение.
- `File not found: {filename}` - указанный файл не найден. Стоит перепроверить имя файла и его расположение
- `The file is empty` - один из входных файлов пуст
- `Invalid character "{original_char}", symbol {char_index} in line {line_index}` - символ не был распознан. Стоит какой-то лишний знак
- `Word duplication, line {line_index}` - дубликат слова (в файле словаря)
- `File lines contain only whitespace characters` - файл состоит только из пробельных символов (т.е. нужной информации не содержат)
- `Inconsistent row lengths` - строки неравной длины (в файле матрицы)
- `It is impossible to place a word in the cell ({i}, {j}) according to the rules` - в клетке (i,j) невозможно поставить слово по правилам (в файле матрицы)

## Пример
Пример работы алгоритма.
Входные данные:
```
*__*__##
###_###*
*_______
###_###_
##*_____
```
Выходные данные:
```
Использованы слова:

 1. айлант
 2. абака
 3. бант
 4. аббатиса
 5. баббит

Заполненный кроссворд:

  а й л а н т . .
  . . . б . . . б
  а б б а т и с а
  . . . к . . . н
  . . б а б б и т
```
