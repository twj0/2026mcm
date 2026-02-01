from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .config import VisualizationConfig, create_output_directories
    from .q1_visualizations import generate_all_q1_visualizations
    from .q2_visualizations import generate_all_q2_visualizations
    from .q3_visualizations import generate_all_q3_visualizations
    from .q4_visualizations import generate_all_q4_visualizations
except ImportError:  # pragma: no cover
    from mcm2026.visualizations.config import VisualizationConfig, create_output_directories
    from mcm2026.visualizations.q1_visualizations import generate_all_q1_visualizations
    from mcm2026.visualizations.q2_visualizations import generate_all_q2_visualizations
    from mcm2026.visualizations.q3_visualizations import generate_all_q3_visualizations
    from mcm2026.visualizations.q4_visualizations import generate_all_q4_visualizations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Generate Q1–Q4 figures (TIFF/EPS/PDF/PNG).')
    parser.add_argument('--data-dir', type=Path, default=Path('.'), help='Project root directory')
    parser.add_argument(
        '--ini',
        type=Path,
        default=None,
        help='Optional visualization ini file path (font/dpi overrides)',
    )
    parser.add_argument('--showcase', action='store_true', help='Also generate appendix-only figures')
    parser.add_argument('--mode', type=str, default='paper', help='paper (4 core figs) or full')
    args = parser.parse_args(argv)

    config = VisualizationConfig.from_ini(args.ini) if args.ini is not None else VisualizationConfig()
    output_structure = create_output_directories(args.data_dir / 'outputs' / 'figures', ['Q1', 'Q2', 'Q3', 'Q4'])

    generate_all_q1_visualizations(
        args.data_dir,
        output_structure['Q1'],
        config,
        showcase=args.showcase,
        mode=str(args.mode),
    )
    generate_all_q2_visualizations(
        args.data_dir,
        output_structure['Q2'],
        config,
        showcase=args.showcase,
        mode=str(args.mode),
    )
    generate_all_q3_visualizations(
        args.data_dir,
        output_structure['Q3'],
        config,
        showcase=args.showcase,
        mode=str(args.mode),
    )
    generate_all_q4_visualizations(
        args.data_dir,
        output_structure['Q4'],
        config,
        showcase=args.showcase,
        mode=str(args.mode),
    )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
