import pandas as pd
import numpy as np
from pathlib import Path

from data_utils import (
    read_wells,
    read_layers,
    read_pumping_test,
    print_table_info,
    DatasetInfo
)


# Определяем корень репозитория
ROOT = Path(__file__).resolve().parents[2]

# Пути к исходным файлам
DATA_DIR = ROOT / "data" / "raw" / "hw01"

wells_path = DATA_DIR / "wells.csv"
layers_path = DATA_DIR / "layers.xlsx"
pumping_path = DATA_DIR / "pumping_test.txt"


# Читаем таблицы
wells = read_wells(wells_path)
layers = read_layers(layers_path)
pumping_test = read_pumping_test(pumping_path)


# Выводим информацию о таблицах
print_table_info("Скважины", wells)
print_table_info("Слои", layers)
print_table_info("Испытание откачкой", pumping_test)


# Создаём объекты DatasetInfo
datasets = [
    DatasetInfo("Скважины", wells),
    DatasetInfo("Слои", layers),
    DatasetInfo("Испытание откачкой", pumping_test)
]

for dataset in datasets:
    print(dataset.describe())

# Проверка пропущенных значений
print("\n--- Пропущенные значения ---")
print(wells.isna().sum())


# Удаление строки с отсутствующим давлением
wells_clean = wells.dropna(subset=["pressure_mpa"]).copy()

print("\nРазмер таблицы после очистки:", wells_clean.shape)


# Проверка корректности данных
assert (wells_clean["radius_m"] > 0).all()
assert (layers["thickness_m"] > 0).all()
assert layers["porosity_fraction"].between(0, 1).all()

print("Все проверки assert пройдены!")


# Сохранение очищенной таблицы
processed_dir = ROOT / "data" / "processed" / "hw01"
processed_dir.mkdir(parents=True, exist_ok=True)

wells_clean.to_csv(processed_dir / "wells_clean.csv", index=False)

print("Очищенная таблица сохранена.")

# Перевод давления из МПа в Па
wells_clean["pressure_pa"] = wells_clean["pressure_mpa"] * 1_000_000

# Разность давлений относительно 12 МПа
wells_clean["pressure_difference_mpa"] = (
    12 - wells_clean["pressure_mpa"]
)

# Относительное изменение в процентах
wells_clean["relative_change_percent"] = (
    wells_clean["pressure_difference_mpa"] / 12 * 100
)

print("\n--- Вычисления давления ---")
print(wells_clean[
    ["pressure_mpa", "pressure_pa",
     "pressure_difference_mpa", "relative_change_percent"]
])

# Статистические характеристики
columns_to_analyze = {
    "pressure_mpa": wells_clean["pressure_mpa"],
    "radius_m": wells_clean["radius_m"],
    "thickness_m": layers["thickness_m"],
    "porosity_fraction": layers["porosity_fraction"]
}

print("\n--- Статистические характеристики ---")

for name, values in columns_to_analyze.items():
    print(f"\n{name}")
    print("Минимум:", values.min())
    print("Максимум:", values.max())
    print("Среднее:", values.mean())

# Проверка координат через sin и cos

theta_rad = wells_clean["azimuth_deg"] * np.pi / 180

wells_clean["x_check"] = wells_clean["radius_m"] * np.cos(theta_rad)
wells_clean["y_check"] = wells_clean["radius_m"] * np.sin(theta_rad)

assert np.allclose(
    wells_clean["x_check"],
    wells_clean["x_m"],
    atol=0.02
)

assert np.allclose(
    wells_clean["y_check"],
    wells_clean["y_m"],
    atol=0.02
)

print("\nКоординаты успешно проверены!")


# Вычисление логарифма радиуса

wells_clean["log_radius"] = np.log(wells_clean["radius_m"])

print("\n--- Логарифм радиуса ---")
print(wells_clean[["radius_m", "log_radius"]])


# Вычисление экспоненты

pumping_test["decay"] = np.exp(-pumping_test["time_h"] / 36)

assert np.isclose(pumping_test["decay"].iloc[0], 1.0)
assert (pumping_test["decay"].diff().dropna() < 0).all()

print("\n--- Последовательность decay ---")
print(pumping_test[["time_h", "decay"]])

# Преобразуем давление в массив NumPy

pressure = wells_clean["pressure_mpa"].to_numpy()

print("\n--- Индексация и срезы ---")

print("Первое давление:", pressure[0])
print("Последнее давление:", pressure[-1])

print("Первые три давления:", pressure[:3])
print("Каждое второе давление:", pressure[::2])


# Логическая маска

mask = pressure < pressure.mean()

print("Логическая маска:", mask)
print("Давления ниже среднего:", pressure[mask])


# Скважины дальше 100 метров

far_wells = wells_clean[wells_clean["radius_m"] > 100]

print("\n--- Скважины дальше 100 м ---")
print(far_wells)

# Создание двумерного массива давлений

pressure_matrix = pumping_test[
    ["boundary_pressure_mpa", "well_pressure_mpa"]
].to_numpy()

print("\n--- pressure_matrix ---")
print("Количество измерений:", pressure_matrix.ndim)
print("Форма массива:", pressure_matrix.shape)
print("Количество элементов:", pressure_matrix.size)
print("Тип данных:", pressure_matrix.dtype)
print(pressure_matrix)


# Создание двух экспериментов

experiment_1 = pressure_matrix
experiment_2 = pressure_matrix + 0.05

pressure_cube = np.stack(
    [experiment_1, experiment_2],
    axis=0
)

print("\n--- pressure_cube ---")
print("Форма массива:", pressure_cube.shape)

print("Первый эксперимент:")
print(pressure_cube[0])

print("Второй эксперимент:")
print(pressure_cube[1])

print("Первая строка первого эксперимента:")
print(pressure_cube[0, 0])

print("Давление в скважине во всех экспериментах:")
print(pressure_cube[:, :, 1])


# Транспонирование

pressure_transposed = pressure_matrix.T

print("\nФорма транспонированного массива:")
print(pressure_transposed.shape)


# Изменение формы массива

flat = pressure_cube.reshape(-1)
restored = flat.reshape(pressure_cube.shape)

assert np.allclose(restored, pressure_cube)

print("Размер одномерного массива:", flat.shape)
print("Восстановление массива прошло успешно!")

# Создание папки для экспортируемых файлов

exports_dir = ROOT / "exports" / "hw01"
exports_dir.mkdir(parents=True, exist_ok=True)


# Повторное сохранение очищенной таблицы со всеми вычислениями

wells_clean.to_csv(
    processed_dir / "wells_clean.csv",
    index=False
)


# Создание сводной таблицы статистики

summary = pd.DataFrame({
    name: {
        "minimum": values.min(),
        "maximum": values.max(),
        "mean": values.mean()
    }
    for name, values in columns_to_analyze.items()
}).T

summary.to_excel(processed_dir / "table_summary.xlsx")

print("\nСводная таблица:")
print(summary)


# Сохранение двумерного массива NPY

np.save(processed_dir / "pressure_matrix.npy", pressure_matrix)

matrix_loaded = np.load(processed_dir / "pressure_matrix.npy")

assert np.allclose(matrix_loaded, pressure_matrix)


# Сохранение трёхмерного массива NPZ

np.savez(
    processed_dir / "pressure_cube.npz",
    pressure_cube=pressure_cube
)

cube_data = np.load(processed_dir / "pressure_cube.npz")
cube_loaded = cube_data["pressure_cube"]

assert np.allclose(cube_loaded, pressure_cube)

cube_data.close()

print("\nПроверка загрузки NPY и NPZ пройдена!")


# Сохранение текстового отчёта

with open(exports_dir / "results.txt", "w", encoding="utf-8") as file:
    file.write(f"pressure_matrix shape: {pressure_matrix.shape}\n")
    file.write(f"pressure_cube shape: {pressure_cube.shape}\n")

    for name, values in columns_to_analyze.items():
        file.write(
            f"{name}: min={values.min()}, "
            f"max={values.max()}, "
            f"mean={values.mean()}\n"
        )

print("Все результаты сохранены!")
