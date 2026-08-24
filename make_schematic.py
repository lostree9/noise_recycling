from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parent
tex = root / "fig_unfolding_tikz.tex"
out_pdf = root / "fig_unfolding_schematic.pdf"
out_png = root / "fig_unfolding_schematic.png"

if shutil.which("pdflatex") is None:
    raise SystemExit("pdflatex is required to build the schematic")

subprocess.run(
    ["pdflatex", "-interaction=nonstopmode", tex.name],
    cwd=root,
    check=True,
    stdout=subprocess.DEVNULL,
)

built = root / "fig_unfolding_tikz.pdf"
built.replace(out_pdf)

# Optional PNG preview for quick visual checks.
if shutil.which("pdftoppm") is not None:
    subprocess.run(
        ["pdftoppm", "-singlefile", "-r", "220", str(out_pdf), str(root / "fig_unfolding_schematic"), "-png"],
        check=True,
        stdout=subprocess.DEVNULL,
    )

print(out_pdf)
if out_png.exists():
    print(out_png)
