
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Пути к файлам
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

pumping_path = ROOT / "data/raw/hw01/pumping_test.txt"
wells_path = ROOT / "data/processed/hw01/wells_clean.csv"

required_figures = [
    ROOT / "figures/hw02/pressure_over_time.svg",
    ROOT / "figures/hw02/pressure_by_radius.svg",
]

output_dir = ROOT / "figures/hw03"
output_dir.mkdir(parents=True, exist_ok=True)

homework_dir = ROOT / "homeworks/hw03"


# ============================================================
# 2. Проверка предыдущих результатов и загрузка таблиц
# ============================================================

for path in required_figures:
    assert path.exists()
    assert path.stat().st_size > 0

pumping_test = pd.read_csv(pumping_path, sep="\t")
wells = pd.read_csv(wells_path)

assert pumping_test.shape == (13, 4)
assert wells.shape[0] == 7

assert not pumping_test.isna().any().any()
assert not wells.isna().any().any()

print("Исходные данные проверены.")


# ============================================================
# 3. Подготовка массивов
# ============================================================

time_h = pumping_test["time_h"].to_numpy()

boundary_pressure_mpa = pumping_test[
    "boundary_pressure_mpa"
].to_numpy()

well_pressure_mpa = pumping_test[
    "well_pressure_mpa"
].to_numpy()

well_id = wells["well_id"].to_numpy(dtype=str)

radius_m = wells["radius_m"].to_numpy()

observation_pressure_mpa = wells[
    "pressure_mpa"
].to_numpy()


# ============================================================
# 4. Сортировка наблюдательных скважин
# ============================================================

order = np.argsort(radius_m)

well_id = well_id[order]
radius_m = radius_m[order]
observation_pressure_mpa = observation_pressure_mpa[order]


# ============================================================
# 5. Расчёт падения давления
# ============================================================

pressure_drop_mpa = (
    boundary_pressure_mpa - well_pressure_mpa
)

observation_drop_mpa = (
    12.0 - observation_pressure_mpa
)


# ============================================================
# 6. Композиция графиков 2 x 2
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 8),
    constrained_layout=True,
)


# ------------------------------------------------------------
# Панель A. Линейный график
# ------------------------------------------------------------

ax = axes[0, 0]

ax.plot(
    time_h,
    boundary_pressure_mpa,
    marker="o",
    label="Boundary pressure",
)

ax.plot(
    time_h,
    well_pressure_mpa,
    marker="s",
    label="Well pressure",
)

ax.set_title("A. Pressure over time")
ax.set_xlabel("Time, h")
ax.set_ylabel("Pressure, MPa")
ax.legend()
ax.grid(True, alpha=0.3)


# ------------------------------------------------------------
# Панель B. Точечная диаграмма
# ------------------------------------------------------------

ax = axes[0, 1]

scatter = ax.scatter(
    radius_m,
    observation_pressure_mpa,
    c=observation_pressure_mpa,
    cmap="viridis",
    s=80,
)

ax.set_title("B. Pressure by radius")
ax.set_xlabel("Radius, m")
ax.set_ylabel("Pressure, MPa")
ax.grid(True, alpha=0.3)

colorbar = fig.colorbar(
    scatter,
    ax=ax,
)

colorbar.set_label("Pressure, MPa")


# ------------------------------------------------------------
# Панель C. Столбчатая диаграмма
# ------------------------------------------------------------

ax = axes[1, 0]

ax.bar(
    well_id,
    observation_drop_mpa,
)

ax.set_title("C. Pressure drop by well")
ax.set_xlabel("Well ID")
ax.set_ylabel("Pressure drop, MPa")
ax.grid(axis="y", alpha=0.3)


# ------------------------------------------------------------
# Панель D. Диаграмма размаха
# ------------------------------------------------------------

ax = axes[1, 1]

ax.boxplot(
    [
        boundary_pressure_mpa,
        well_pressure_mpa,
    ],
    tick_labels=[
        "Boundary",
        "Well",
    ],
)

ax.set_title("D. Pressure distributions")
ax.set_xlabel("Pressure type")
ax.set_ylabel("Pressure, MPa")
ax.grid(axis="y", alpha=0.3)


# ------------------------------------------------------------
# Сохранение композиции
# ------------------------------------------------------------

fig.savefig(
    output_dir / "pressure_overview.png",
    dpi=300,
)

fig.savefig(
    output_dir / "pressure_overview.svg",
)

plt.close(fig)


# ============================================================
# 7. Сравнение boxplot и violinplot
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 5),
    constrained_layout=True,
)

pressure_data = [
    boundary_pressure_mpa,
    well_pressure_mpa,
]

labels = [
    "Boundary",
    "Well",
]


# ------------------------------------------------------------
# Левая панель. Boxplot
# ------------------------------------------------------------

ax = axes[0]

ax.boxplot(
    pressure_data,
    tick_labels=labels,
)

ax.set_title("Boxplot")
ax.set_xlabel("Pressure type")
ax.set_ylabel("Pressure, MPa")
ax.grid(axis="y", alpha=0.3)


# ------------------------------------------------------------
# Правая панель. Violinplot
# ------------------------------------------------------------

ax = axes[1]

ax.violinplot(
    pressure_data,
    showmedians=True,
)

ax.set_xticks(
    [1, 2],
    labels=labels,
)

ax.set_title("Violin plot")
ax.set_xlabel("Pressure type")
ax.set_ylabel("Pressure, MPa")
ax.grid(axis="y", alpha=0.3)


# ------------------------------------------------------------
# Сохранение сравнения
# ------------------------------------------------------------

fig.savefig(
    output_dir / "distributions.png",
    dpi=300,
)

fig.savefig(
    output_dir / "distributions.svg",
)

plt.close(fig)

print("Графики успешно сохранены.")


# ============================================================
# 8. Сохранение данных для ДЗ №4
# ============================================================

output_path = homework_dir / "viz_data.npz"

np.savez(
    output_path,
    time_h=time_h,
    boundary_pressure_mpa=boundary_pressure_mpa,
    well_pressure_mpa=well_pressure_mpa,
    pressure_drop_mpa=pressure_drop_mpa,
    well_id=well_id,
    radius_m=radius_m,
    observation_pressure_mpa=observation_pressure_mpa,
    observation_drop_mpa=observation_drop_mpa,
)


# ============================================================
# 9. Проверка сохранённого архива
# ============================================================

expected_data = {
    "time_h": time_h,
    "boundary_pressure_mpa": boundary_pressure_mpa,
    "well_pressure_mpa": well_pressure_mpa,
    "pressure_drop_mpa": pressure_drop_mpa,
    "well_id": well_id,
    "radius_m": radius_m,
    "observation_pressure_mpa": observation_pressure_mpa,
    "observation_drop_mpa": observation_drop_mpa,
}

with np.load(output_path) as data:
    assert set(data.files) == set(expected_data.keys())

    for key, expected in expected_data.items():
        loaded = data[key]

        assert loaded.shape == expected.shape
        assert np.array_equal(loaded, expected)

print("Архив viz_data.npz успешно сохранён и проверен.")


# ============================================================
# 10. Проверка итоговых файлов
# ============================================================

required_outputs = [
    output_dir / "pressure_overview.png",
    output_dir / "pressure_overview.svg",
    output_dir / "distributions.png",
    output_dir / "distributions.svg",
    output_path,
]

for path in required_outputs:
    assert path.exists()
    assert path.stat().st_size > 0

print("ДЗ №3 успешно выполнено.")
