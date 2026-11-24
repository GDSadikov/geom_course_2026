
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


# Пути к файлам
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "hw01"
PROCESSED_DIR = ROOT / "data" / "processed" / "hw01"
EXPORTS_DIR = ROOT / "exports" / "hw01"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


# 1. Чтение исходных данных

wells = read_wells(DATA_DIR / "wells.csv")
layers = read_layers(DATA_DIR / "layers.xlsx")
pumping_test = read_pumping_test(DATA_DIR / "pumping_test.txt")

print("\n========== ИСХОДНЫЕ ДАННЫЕ ==========")

print_table_info("Скважины", wells)
print_table_info("Слои", layers)
print_table_info("Испытание откачкой", pumping_test)


# 2. Информация о наборах данных

datasets = [
    DatasetInfo("Скважины", wells),
    DatasetInfo("Слои", layers),
    DatasetInfo("Испытание откачкой", pumping_test)
]

print("\n========== ИНФОРМАЦИЯ О НАБОРАХ ==========")

for dataset in datasets:
    print(dataset.describe())


# 3. Очистка и проверка данных

print("\n========== ОЧИСТКА ДАННЫХ ==========")

print("Пропущенные значения:")
print(wells.isna().sum())

wells_clean = wells.dropna(subset=["pressure_mpa"]).copy()

print("Размер таблицы после очистки:", wells_clean.shape)

assert (wells_clean["radius_m"] > 0).all()
assert (layers["thickness_m"] > 0).all()
assert layers["porosity_fraction"].between(0, 1).all()

print("Все проверки корректности данных пройдены!")


# 4. Вычисления давления

wells_clean["pressure_pa"] = (
    wells_clean["pressure_mpa"] * 1_000_000
)

wells_clean["pressure_difference_mpa"] = (
    12 - wells_clean["pressure_mpa"]
)

wells_clean["relative_change_percent"] = (
    wells_clean["pressure_difference_mpa"] / 12 * 100
)

print("\n========== ВЫЧИСЛЕНИЯ ДАВЛЕНИЯ ==========")

print(wells_clean[
    ["pressure_mpa", "pressure_pa",
     "pressure_difference_mpa", "relative_change_percent"]
].round(3))


# 5. Статистические характеристики

columns_to_analyze = {
    "pressure_mpa": wells_clean["pressure_mpa"],
    "radius_m": wells_clean["radius_m"],
    "thickness_m": layers["thickness_m"],
    "porosity_fraction": layers["porosity_fraction"]
}

summary = pd.DataFrame({
    name: {
        "minimum": values.min(),
        "maximum": values.max(),
        "mean": values.mean()
    }
    for name, values in columns_to_analyze.items()
}).T

print("\n========== СТАТИСТИКА ==========")
print(summary.round(3))


# 6. Проверка координат через sin и cos

theta_rad = wells_clean["azimuth_deg"] * np.pi / 180

wells_clean["x_check"] = (
    wells_clean["radius_m"] * np.cos(theta_rad)
)

wells_clean["y_check"] = (
    wells_clean["radius_m"] * np.sin(theta_rad)
)

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


# 7. Логарифм радиуса

wells_clean["log_radius"] = np.log(wells_clean["radius_m"])

print("\n========== ЛОГАРИФМ РАДИУСА ==========")
print(wells_clean[["radius_m", "log_radius"]].round(3))


# 8. Экспоненциальная функция затухания

pumping_test["decay"] = np.exp(-pumping_test["time_h"] / 36)

assert np.isclose(pumping_test["decay"].iloc[0], 1.0)
assert (pumping_test["decay"].diff().dropna() < 0).all()

print("\n========== ЭКСПОНЕНЦИАЛЬНОЕ ЗАТУХАНИЕ ==========")
print(pumping_test[["time_h", "decay"]].round(4))


# 9. Индексация, срезы и логические маски NumPy

pressure = wells_clean["pressure_mpa"].to_numpy()

print("\n========== ИНДЕКСАЦИЯ И СРЕЗЫ ==========")

print("Первое давление:", pressure[0])
print("Последнее давление:", pressure[-1])
print("Первые три давления:", pressure[:3])
print("Каждое второе давление:", pressure[::2])

mask = pressure < pressure.mean()

print("\nЛогическая маска:", mask)
print("Давления ниже среднего:", pressure[mask])

far_wells = wells_clean[wells_clean["radius_m"] > 100]

print("\nСкважины дальше 100 метров:")
print(far_wells["well_id"].to_list())


# 10. Создание двумерного массива NumPy

pressure_matrix = pumping_test[
    ["boundary_pressure_mpa", "well_pressure_mpa"]
].to_numpy()

print("\n========== ДВУМЕРНЫЙ МАССИВ ==========")
print("Форма массива:", pressure_matrix.shape)
print("Количество размерностей:", pressure_matrix.ndim)
print("Количество элементов:", pressure_matrix.size)
print("Тип данных:", pressure_matrix.dtype)


# 11. Создание трёхмерного массива NumPy

experiment_1 = pressure_matrix
experiment_2 = pressure_matrix + 0.05

pressure_cube = np.stack(
    [experiment_1, experiment_2],
    axis=0
)

print("\n========== ТРЁХМЕРНЫЙ МАССИВ ==========")
print("Форма массива:", pressure_cube.shape)

print("Первая строка первого эксперимента:")
print(pressure_cube[0, 0])

print("Форма среза давления по скважине:", pressure_cube[:, :, 1].shape)

# 12. Транспонирование и изменение формы

pressure_transposed = pressure_matrix.T

flat = pressure_cube.reshape(-1)
restored = flat.reshape(pressure_cube.shape)

assert np.allclose(restored, pressure_cube)

print("\n========== ИЗМЕНЕНИЕ ФОРМЫ МАССИВОВ ==========")
print("Форма транспонированного массива:", pressure_transposed.shape)
print("Форма одномерного массива:", flat.shape)
print("Восстановление массива прошло успешно!")


# 13. Сохранение результатов

wells_clean.to_csv(
    PROCESSED_DIR / "wells_clean.csv",
    index=False
)

summary.to_excel(PROCESSED_DIR / "table_summary.xlsx")

np.save(
    PROCESSED_DIR / "pressure_matrix.npy",
    pressure_matrix
)

np.savez(
    PROCESSED_DIR / "pressure_cube.npz",
    pressure_cube=pressure_cube
)


# 14. Повторная загрузка и проверка массивов

matrix_loaded = np.load(PROCESSED_DIR / "pressure_matrix.npy")

assert np.allclose(matrix_loaded, pressure_matrix)

cube_data = np.load(PROCESSED_DIR / "pressure_cube.npz")
cube_loaded = cube_data["pressure_cube"]

assert np.allclose(cube_loaded, pressure_cube)

cube_data.close()

print("\nПроверка сохранения и повторной загрузки NPY и NPZ пройдена!")


# 15. Создание текстового отчёта

with open(
    EXPORTS_DIR / "results.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(f"pressure_matrix shape: {pressure_matrix.shape}\n")
    file.write(f"pressure_cube shape: {pressure_cube.shape}\n")

    for name, values in columns_to_analyze.items():
        file.write(
            f"{name}: min={values.min()}, "
            f"max={values.max()}, "
            f"mean={values.mean()}\n"
        )

print("\n========== ЗАВЕРШЕНО ==========")
print("Все вычисления выполнены.")
print("Все проверки пройдены.")
print("Результаты сохранены.")
