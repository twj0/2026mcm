from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple
import warnings
import zlib

import matplotlib.pyplot as plt
from matplotlib import cycler
from matplotlib.offsetbox import AnchoredText


@dataclass(frozen=True)
class VisualizationConfig:
    # 首选字体是新罗马
    font_family: str = "Times New Roman"
    # 如果新罗马不可用(电脑没安装新罗马)，则使用Arial
    font_fallback: str = "Arial"
    # 图片的分辨率，默认是300dpi
    dpi: int = 300
    style: str = "modern"

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
        palette = list(self.get_color_cycle())
        plt.rcParams.update(
            {
                "font.size": 9.5,
                "axes.titlesize": 10.5,
                "axes.labelsize": 9.5,
                "xtick.labelsize": 8.5,
                "ytick.labelsize": 8.5,
                "legend.fontsize": 8.5,
                "axes.titlepad": 8.0,
                "axes.labelpad": 6.0,
                "font.family": "serif",
                "font.serif": [self.font_family, self.font_fallback],
                "mathtext.fontset": "stix",
                "text.color": "#111827",
                "axes.labelcolor": "#111827",
                "xtick.color": "#374151",
                "ytick.color": "#374151",
                "axes.edgecolor": "#111827",
                "axes.facecolor": "#ffffff",
                "figure.facecolor": "#ffffff",
                "figure.dpi": self.dpi,
                "savefig.dpi": self.dpi,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.02,
                "savefig.transparent": False,
                "legend.handlelength": 1.4,
                "legend.handletextpad": 0.5,
                "legend.borderaxespad": 0.3,
                "axes.grid": True,
                "axes.axisbelow": True,
                "grid.alpha": 0.25,
                "grid.color": "#94a3b8",
                "grid.linewidth": 0.6,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.linewidth": 0.8,
                "lines.linewidth": 1.8,
                "lines.markersize": 6,
                "axes.prop_cycle": cycler(color=palette),
                "legend.frameon": False,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )

    def get_named_colors(self) -> Dict[str, str]:
        return {
            "text": "#111827",
            "muted": "#64748b",
            "primary": "#2563eb",
            "secondary": "#0ea5e9",
            "success": "#16a34a",
            "warning": "#f59e0b",
            "danger": "#dc2626",
        }

    def get_mechanism_colors(self) -> Dict[str, str]:
        return {
            "percent": "#2563eb",
            "rank": "#f97316",
            "percent_judge_save": "#16a34a",
            "percent_sqrt": "#dc2626",
            "percent_log": "#7c3aed",
            "percent_cap": "#db2777",
            "dynamic_weight": "#64748b",
        }

    def get_color(self, name: str) -> str:
        s = str(name)
        mech = self.get_mechanism_colors()
        if s in mech:
            return mech[s]
        named = self.get_named_colors()
        if s in named:
            return named[s]

        cycle = self.get_color_cycle()
        if len(cycle) == 0:
            return named["text"]
        idx = int(zlib.adler32(s.encode("utf-8")) % len(cycle))
        return str(cycle[idx])

    def get_color_cycle(self) -> Tuple[str, ...]:
        base = self.get_named_colors()
        mech = self.get_mechanism_colors()
        cycle = [
            base["primary"],
            base["secondary"],
            mech["rank"],
            mech["percent_judge_save"],
            mech["percent_log"],
            mech["percent_cap"],
            mech["dynamic_weight"],
            mech["percent_sqrt"],
        ]
        return tuple(cycle)

    def get_cmap(self, name: str):
        key = str(name).strip().lower()
        if key in {"div", "diverging", "diverge"}:
            return plt.get_cmap("RdYlBu_r")
        if key in {"corr", "correlation"}:
            return plt.get_cmap("RdBu_r")
        if key in {"seq", "sequential"}:
            return plt.get_cmap("viridis")
        if key in {"heat", "heatmap"}:
            return plt.get_cmap("mako") if "mako" in plt.colormaps() else plt.get_cmap("viridis")
        return plt.get_cmap(key)

    def callout_bbox(self, *, kind: str = "default") -> Dict[str, object]:
        k = str(kind)
        if k == "note":
            fc = "#f8fafc"
            ec = "#cbd5e1"
        elif k == "warn":
            fc = "#fffbeb"
            ec = "#f59e0b"
        else:
            fc = "#f8fafc"
            ec = "#cbd5e1"
        return {
            "boxstyle": "round,pad=0.25",
            "facecolor": fc,
            "edgecolor": ec,
            "linewidth": 0.8,
            "alpha": 0.95,
        }

    def add_panel_label(self, ax: plt.Axes, label: str) -> None:
        ax.text(
            0.0,
            1.02,
            str(label),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
        )

    def add_callout(self, ax: plt.Axes, text: str, *, loc: str = "upper left", kind: str = "note") -> None:
        at = AnchoredText(
            str(text),
            loc=str(loc),
            prop={"size": 8.5},
            frameon=True,
            borderpad=0.35,
        )
        patch = at.patch
        patch.set_boxstyle(self.callout_bbox(kind=kind)["boxstyle"])
        patch.set_facecolor(self.callout_bbox(kind=kind)["facecolor"])
        patch.set_edgecolor(self.callout_bbox(kind=kind)["edgecolor"])
        patch.set_linewidth(float(self.callout_bbox(kind=kind)["linewidth"]))
        patch.set_alpha(float(self.callout_bbox(kind=kind)["alpha"]))
        ax.add_artist(at)

    def get_figure_size(self, name: str) -> Tuple[float, float]:
        sizes = {
            "single_column": (6.0, 4.0),
            "double_column": (12.0, 4.5),
            "large_figure": (12.0, 8.0),
            "panel_2x2": (12.0, 9.0),
        }
        return sizes.get(name, sizes["single_column"])


def create_output_directories(base_output_dir: Path, questions: Iterable[str]) -> Dict[str, Dict[str, Path]]:
    out: Dict[str, Dict[str, Path]] = {}
    for q in questions:
        q_upper = q.upper()
        q_lower = q.lower()
        tiff_dir = base_output_dir / q_lower / "tiff"
        eps_dir = base_output_dir / q_lower / "eps"
        pdf_dir = base_output_dir / q_lower / "pdf"
        png_dir = base_output_dir / q_lower / "png"
        tiff_dir.mkdir(parents=True, exist_ok=True)
        eps_dir.mkdir(parents=True, exist_ok=True)
        pdf_dir.mkdir(parents=True, exist_ok=True)
        png_dir.mkdir(parents=True, exist_ok=True)
        out[q_upper] = {"tiff": tiff_dir, "eps": eps_dir, "pdf": pdf_dir, "png": png_dir}
    return out


def save_figure_with_config(
    fig: plt.Figure,
    filename_stem: str,
    output_dirs: Dict[str, Path],
    config: VisualizationConfig,
) -> None:
    tiff_path = output_dirs["tiff"] / f"{filename_stem}.tiff"
    eps_path = output_dirs["eps"] / f"{filename_stem}.eps"
    pdf_path = output_dirs.get("pdf", output_dirs["eps"]) / f"{filename_stem}.pdf"
    png_path = output_dirs.get("png", output_dirs["tiff"]) / f"{filename_stem}.png"

    fig.savefig(tiff_path, dpi=config.dpi, bbox_inches="tight", format="tiff")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The PostScript backend does not support transparency.*",
            category=UserWarning,
        )
        fig.savefig(eps_path, bbox_inches="tight", format="eps")

    fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
    fig.savefig(png_path, dpi=config.dpi, bbox_inches="tight", format="png")
    plt.close(fig)
