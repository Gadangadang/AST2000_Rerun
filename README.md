# AST2000
This repo contains a rerun of the project I did in the course AST2000 a few years back: the UiO
AST2000 spacecraft project (Parts 1-9), redone with `ast2000tools`.

## Running

    .venv/bin/python run_all.py                # every finished part, in order (~6 min)
    .venv/bin/python run_all.py --papers       # ... and compile the papers
    .venv/bin/python run_all.py --seed 81107   # someone else's solar system (or --username abc123)

Each part writes its figures to `Project N/figures/` and the numbers its paper quotes to
`Project N/paper/results.tex`. A new seed overwrites all of these; rerun without `--seed`
to get ours back.

### Data that is not in the repository

- `Project 4/himmelkule.npy`: the sky sphere for the orientation pictures. It is the same
  for every seed.
- `Project 6/spectra/`: the course's atmosphere spectra for all 100 seeds
  (`spectrum_seedXX.parquet`, XX = the last two digits of the seed) and the noise
  (`sigma_noise.parquet`): 2.8 GB of float32, zstd with byte-stream-split (lossless).
  They live on this repository's GitHub release `spectra`, not in git. Part 6 downloads the
  one a seed needs automatically (with the `gh` CLI, logged in), and
  `python spectrum.py --download` (in `Project 6/`) fetches all of them. Course files
  dropped in `Project 6/` (`.txt` or `.zip`) are converted on first use, or all at once with
  `python spectrum.py --import`, which deletes the originals. Without a seed's spectrum,
  Part 6B is skipped and 6C takes the molecular weight from the drag probe alone.

## Papers

One paper per part, in `Project N/paper/partN.tex` (REVTeX; shared style and bibliography in
`common/paper/`). The text never contains a hand-copied result: every number is a
`\val{key}` filled in from `results.tex`, so rerunning a part (or `run_all.py --seed ...`)
and recompiling updates the paper.

    .venv/bin/python build_papers.py       # all papers -> Project N/paper/partN.pdf
    .venv/bin/python build_papers.py 2 5   # only some

Run the parts first. `--quick` skips some slow studies, so their figures show up as
placeholder boxes, and missing numbers as a red `??key`.
