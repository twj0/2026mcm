from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class VisualizationConfig:
    # 首选字体是新罗马
    font_family: str = "Times New Roman"
    # 如果新罗马不可用(电脑没安装新罗马)，则使用Arial
    font_fallback: str = "Arial"
    # 图片的分辨率，默认是300dpi
    dpi: int = 300

    @classmethod
    def from_ini(cls, ini_path: Path) -> "VisualizationConfig":
        if not ini_path.exists():
            return cls()

        parser = configparser.ConfigParser()
        parser.read(ini_path, encoding="utf-8")

        font_family = parser.get("display", "font_family", fallback=cls.font_family)
        font_fallback = parser.get("display", "font_fallback", fallback=cls.font_fallback)
        dpi = parser.getint("display", "dpi", fallback=cls.dpi)
        return cls(font_family=font_family, font_fallback=font_fallback, dpi=dpi)

    def apply_matplotlib_style(self) -> None:
        plt.rcParams.update(
            {
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 11,
                "font.family": "serif",
                "font.serif": [self.font_family, self.font_fallback],
                "figure.dpi": self.dpi,
            }
        )

    def get_figure_size(self, name: str) -> Tuple[int, int]:
        sizes = {
            "single_column": (12, 8),
            "double_column": (16, 8),
            "large_figure": (16, 12),
        }
        return sizes.get(name, sizes["single_column"])


def create_output_directories(base_output_dir: Path, questions: Iterable[str]) -> Dict[str, Dict[str, Path]]:
    out: Dict[str, Dict[str, Path]] = {}
    for q in questions:
        q_upper = q.upper()
        q_lower = q.lower()
        tiff_dir = base_output_dir / q_lower / "tiff"
        eps_dir = base_output_dir / q_lower / "eps"
        tiff_dir.mkdir(parents=True, exist_ok=True)
        eps_dir.mkdir(parents=True, exist_ok=True)
        out[q_upper] = {"tiff": tiff_dir, "eps": eps_dir}
    return out


def save_figure_with_config(
    fig: plt.Figure,
    filename_stem: str,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig,
) -> None:
    tiff_path = output_dirs["tiff"] / f"{filename_stem}.tiff"
    eps_path = output_dirs["eps"] / f"{filename_stem}.eps"

    fig.savefig(tiff_path, dpi=config.dpi, bbox_inches="tight", format="tiff")
    fig.savefig(eps_path, bbox_inches="tight", format="eps")
    plt.close(fig)
