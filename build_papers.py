"""
Compile the papers, one per part, into "Project N/paper/partN.pdf".

    .venv/bin/python build_papers.py          # every part that has a paper
    .venv/bin/python build_papers.py 1 3      # only some parts
    .venv/bin/python build_papers.py --clean  # also remove LaTeX's auxiliary files afterwards

Each paper reads its figures from "Project N/figures/" and its numbers from
"Project N/paper/results.tex". Both are written by pro_N.py (or run_all.py),
so run the parts first; a number that is missing shows up as a red ??key in the
PDF, and a missing figure as a box saying which script makes it.
Needs a TeX installation with latexmk (TeX Live or MacTeX).
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))


def papers(parts=None):
    """(part, folder, tex file) for every paper in the repo, or only the given parts."""
    found = []
    for tex in sorted(glob.glob(os.path.join(REPO, "Project *", "paper", "part*.tex"))):
        part = int(os.path.basename(os.path.dirname(os.path.dirname(tex))).split()[-1])
        if parts is None or part in parts:
            found.append((part, os.path.dirname(tex), os.path.basename(tex)))
    return found


def missing_values(folder, tex):
    """Keys used with \\val{} in the paper that results.tex does not define."""
    import re
    text = re.sub(r"(?<!\\)%.*", "", open(os.path.join(folder, tex)).read())  # Drop comments
    used = set(re.findall(r"\\val\{([^}]+)\}", text))
    results = os.path.join(folder, "results.tex")
    defined = set(re.findall(r"\\setvalue\{([^}]+)\}", open(results).read())) if os.path.exists(results) else set()
    return sorted(used - defined)


def build(folder, tex, clean):
    command = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-quiet", tex]
    result = subprocess.run(command, cwd=folder, capture_output=True, text=True)
    if clean:
        subprocess.run(["latexmk", "-c", tex], cwd=folder, capture_output=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parts", type=int, nargs="*", help="which parts (default: all with a paper)")
    parser.add_argument("--clean", action="store_true", help="remove auxiliary files after compiling")
    args = parser.parse_args()
    if shutil.which("latexmk") is None:
        sys.exit("latexmk not found: install TeX Live / MacTeX to compile the papers")

    failed = []
    for part, folder, tex in papers(args.parts or None):
        missing = missing_values(folder, tex)
        result = build(folder, tex, args.clean)
        pdf = os.path.join(folder, tex.replace(".tex", ".pdf"))
        if result.returncode == 0:
            note = f"  ({len(missing)} values missing: {', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}; "\
                   f"rerun pro_{part}.py)" if missing else ""
            print(f"Part {part}: {os.path.relpath(pdf, REPO)}{note}")
        else:
            failed.append(part)
            log = (result.stdout + result.stderr).splitlines()
            errors = [line for line in log if line.startswith("!") or "Error" in line][:10]
            print(f"Part {part}: FAILED\n  " + "\n  ".join(errors or log[-15:]))
    if failed:
        sys.exit(f"papers failed: {failed}")


if __name__ == "__main__":
    main()
