# AST2000
This repo contains a rerun of the project I did in the course AST2000 a few years back: the UiO
AST2000 spacecraft project (Parts 1-9), redone with `ast2000tools`.

The project sends a spacecraft from its home planet to another planet in a randomly generated
solar system, and lands it. Each student's solar system is different: a username gives a seed,
and the seed decides the star, the planets and every measurement. Everything is built from first
principles: a rocket engine from a box of gas particles, orbits from Newton's law of gravity,
navigation from pictures of the sky, an atmosphere from a noisy spectrum. Then `ast2000tools`
checks each step by flying it.

This rerun aims to do it properly this time:

- **Reproducible.** One command runs every part in order. Each part feeds the next: the
  launch, the orbits, the orientation software, the journey and the atmosphere model.
- **Any solar system.** `--seed` runs someone else's system end to end. The destination, launch
  window and landing gear are chosen automatically, and the analysis notices when the data
  disagree (for example, a gas the spectrum missed).
- **Papers that cannot go stale.** Each part has a short paper. No number in them is typed by
  hand: they are all filled in from the latest run.
- **Figures for everything.** About 110 of them, each one there to explain a step. They are in
  `Project N/figures/`, and the compiled papers in `Project N/paper/`, as of the last run for our seed.

| | |
|:-:|:-:|
| ![The solar system](Project%202/figures/a3_system_overview.png) | ![The journey](Project%205/figures/c1_flight.png) |
| *Part 2: the solar system of seed 42716, a 1.19 M☉ star and 8 planets* | *Part 5: the real flight to planet 1, with two course corrections of a few cm/s* |
| ![Spectral lines](Project%206/figures/b2_line_fits.png) | ![The landing](Project%207/figures/c2_descent.png) |
| *Part 6: every spectral line fitted by χ²; four lines, from three gases, are real* | *Part 7: the landing, our simulator against `ast2000tools`* |

## The parts

The results are for our seed, 42716 (username `09012023`).

| Part | What it does | Result |
|---|---|---|
| 1 | Simulate a box of hot gas particles, build a rocket engine from many such boxes, and launch | Launch verified by the API, using 378 kg of the 388 kg of fuel loaded |
| 2 | Orbits of the 8 planets, analytic and numerical; Kepler's laws; the star's wobble; radial velocity and transit curves; relativistic perihelion precession | Planet mass from the radial-velocity fit within 0.05% of the true value |
| 3 | Habitable zone, choice of destination, launch windows, and a launch from any site at any time | Destination planet 1, a rocky planet at 281 K; all 8 generalised launches verified |
| 4 | Onboard orientation software: pointing from sky pictures, velocity from the Doppler shift of two stars, position by trilateration | Manual orientation passed the API's check |
| 5 | Mission planner (Lambert solver, porkchop plot), then the real flight with course corrections and orbit injection | 133 days, 3.16 km/s as planned; nearly circular orbit (e = 4.6·10⁻⁴) |
| 6 | Low orbit; χ² search of a 10-million-point spectrum for the atmosphere's gases; density and temperature model checked with a drag probe; landing site | H₂O, CH₄ and CO found, O₂ marginal; molecular weight within 0.06% of the API's hidden value |
| 7 | Terminal velocity, parachute and thrusters; planning and flying the landing. Plus the star: HR diagram and core model | Touchdown at 1.97 m/s, 0.7 km from the chosen site |
| 8, 9 | Special and general relativity | Not done yet |

Each part folder holds the assignment text (`partN.pdf`) and `pro_N.py`, which runs the whole part.
Each challenge is its own module (for example `Project 6/spectrum.py`), and each module can also
be run on its own.

## Setup

    python3.9 -m venv .venv
    .venv/bin/pip install -r requirements.txt

The papers also need LaTeX: `latexmk` and REVTeX, which TeX Live and MacTeX include.

## Running

    .venv/bin/python run_all.py                # every finished part, in order (~5 min)
    .venv/bin/python run_all.py --papers       # ... and compile the papers
    .venv/bin/python run_all.py --seed 81107   # someone else's solar system (or --username abc123)
    .venv/bin/python run_all.py --parts 6 7    # only some parts (they need the earlier parts' outputs)
    .venv/bin/python run_all.py --quick        # skip the slowest side studies

To run a single part, go into its folder and run its script, e.g.
`cd "Project 5" && ../.venv/bin/python pro_5.py`.

Each part writes:
- its figures to `Project N/figures/`;
- the numbers its paper quotes to `Project N/paper/results.tex`;
- whatever the next part needs, such as missions saved after launch, orientation and orbit
  injection, or the atmosphere model.

A new seed overwrites all of these. To get ours back, rerun without `--seed`. The username is
set in `common/config.py`.

### Data that is not in the repository

These files are too large for git, so they are on this repository's GitHub release
[Course data](../../releases/tag/spectra) instead. The code downloads what it needs from there
(`common/release.py`), over HTTPS or with the `gh` CLI.

- `Project 4/himmelkule.npy`: the course's sky sphere for the orientation pictures (126 MB). It is
  the same for every seed. Part 4 downloads it the first time it runs.
- `XMLs.zip`: our run's files for the MCAst/SSView viewer (see below), 130 MB unzipped. Every run
  writes them again.
- `Project 6/spectra/`: the course's atmosphere spectra for all 100 seeds, plus the noise. They
  are stored as `spectrum_seedXX.parquet`, where XX is the last two digits of the seed, and
  `sigma_noise.parquet`.
  - They take 2.8 GB as float32, compressed losslessly with zstd and byte-stream-split.
  - Part 6 downloads the one a seed needs automatically. Running
    `python spectrum.py --download` (in `Project 6/`) fetches all of them.
  - Course files dropped into `Project 6/` (`.txt` or `.zip`) are converted on first use. To
    convert them all at once, run `python spectrum.py --import`, which deletes the originals.
  - Without a seed's spectrum, Part 6B is skipped and 6C takes the molecular weight from the
    drag probe alone.

### Watching the flights

`ast2000tools` writes XML files of the orbits, the journey and the landing to `XMLs/`. If the
course's MCAst/SSView viewer is unpacked in `MCAst/`, the parts copy those files into its data
folders, so they can be watched in 3D.

## Papers

There is one paper per part, in `Project N/paper/partN.tex`. They use REVTeX, with a shared
style and bibliography in `common/paper/`.

The text never contains a hand-copied result. Every number is a `\val{key}`, filled in from
`results.tex`, so rerunning a part (or `run_all.py --seed ...`) and recompiling updates the paper.

    .venv/bin/python build_papers.py       # all papers -> Project N/paper/partN.pdf
    .venv/bin/python build_papers.py 2 5   # only some

Run the parts first. A figure that has not been made yet shows up as a placeholder box, and a
missing number as a red `??key`.

## Layout

    run_all.py           every part in order (--seed, --parts, --quick, --papers)
    build_papers.py      compiles the papers
    common/              shared code: the username and seed (config.py), integrators, plot style,
                         the results pipeline for the papers (report.py), and the part runner
    common/paper/        the LaTeX style and the bibliography
    Project N/           part N: pro_N.py, one module per challenge, paper/, and the assignment text

## Credits

The project and its assignment texts are from AST2000 at the University of Oslo. The course
provides `ast2000tools`, the sky sphere and the spectra.
