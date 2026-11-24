
from pathlib import Path

import pandas as pd

from plot_utils import (
    plot_pressure_over_time,
    plot_pressure_by_radius,
)


def main():
    project_root = Path(__file__).resolve().parents[2]

    pumping_test_path = (
        project_root / "data" / "raw" / "hw01" / "pumping_test.txt"
    )

    wells_path = (
        project_root / "data" / "processed" / "hw01" / "wells_clean.csv"
    )

    figures_dir = project_root / "figures" / "hw02"
    figures_dir.mkdir(parents=True, exist_ok=True)

    pumping_test = pd.read_csv(pumping_test_path, sep="\t")
    wells_clean = pd.read_csv(wells_path)

    plot_pressure_over_time(pumping_test, figures_dir)
    plot_pressure_by_radius(wells_clean, figures_dir)

    assert (figures_dir / "pressure_over_time.png").exists()
    assert (figures_dir / "pressure_over_time.svg").exists()
    assert (figures_dir / "pressure_by_radius.png").exists()
    assert (figures_dir / "pressure_by_radius.svg").exists()

    print("Графики успешно созданы!")


if __name__ == "__main__":
    main()
