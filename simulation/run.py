"""运行：python run.py；仅重绘图件：python run.py --figures-only。"""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SIMULATIONS = [
    ["experiment.py"], ["audit_single_origins.py"],
    ["closed_loop.py", "calibration"], ["closed_loop.py", "validation"],
    ["extension.py", "calibrate"], ["extension.py", "density"],
    ["extension.py", "geometry"], ["extension.py", "structure"],
    ["extension.py", "timing"], ["refinement.py"],
]
FIGURES = [["plot_results.py"], ["plot_revision_figures.py"],
           ["plot_closed_loop.py"], ["plot_extension.py"]]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-only", action="store_true", help="使用包内结果重绘全部图件")
    args = parser.parse_args()
    tasks = FIGURES if args.figures_only else SIMULATIONS + FIGURES
    for command in tasks:
        print("Running: " + " ".join(command), flush=True)
        subprocess.run([sys.executable, *command], cwd=ROOT, check=True)

if __name__ == "__main__":
    main()
