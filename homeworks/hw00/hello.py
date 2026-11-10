import sys

# Импортируем библиотеки, которые мы установили в виртуальное окружение
import numpy
import pandas
import matplotlib

# Выводим версии, запрашивая их напрямую у программ
print(f"Версия Python: {sys.version.split()[0]}")
print(f"Версия NumPy: {numpy.__version__}")
print(f"Версия Pandas: {pandas.__version__}")
print(f"Версия Matplotlib: {matplotlib.__version__}")