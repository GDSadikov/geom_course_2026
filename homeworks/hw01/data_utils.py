
import pandas as pd


def read_wells(path):
    """Читает данные о наблюдательных скважинах из CSV."""
    return pd.read_csv(path)


def read_layers(path):
    """Читает данные о слоях из Excel."""
    return pd.read_excel(path)


def read_pumping_test(path):
    """Читает данные испытания из текстового файла."""
    return pd.read_csv(path, sep="\t")


def print_table_info(name, table):
    """Выводит основную информацию о таблице."""
    print(f"\n--- {name} ---")
    print("Размер:", table.shape)
    print("Столбцы:", table.columns)
    print("Типы данных:")
    print(table.dtypes)
    print("Первые пять строк:")
    print(table.head())


class DatasetInfo:
    def __init__(self, name, table):
        self.name = name
        self.rows, self.columns = table.shape

    def describe(self):
        return f"{self.name}: {self.rows} строк, {self.columns} столбцов"
