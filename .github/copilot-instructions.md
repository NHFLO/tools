# NHFLO Copilot instructions for GPT-5.5

Keep this file identical in `NHFLO/data`, `NHFLO/tools`, and `NHFLO/models`. Repository README files, `pyproject.toml`, tests, and local conventions are the source of truth when more specific.

## GPT-5.5 operating rules

- Inspect before asserting; cite files or command output when reporting non-obvious findings.
- Be context-frugal: read targeted files, avoid broad dumps, and do not narrate routine steps.
- Make small, complete changes. Do not touch unrelated dirty worktree changes.
- Preserve scientific meaning: units, CRS, sign conventions, layer numbering, time discretization, and MODFLOW semantics.
- Never commit secrets, confidential data, generated model output, downloaded binaries, caches, notebook outputs, or absolute local paths.

## Repository boundaries

- `NHFLO/models`: model applications under `modelscripts\<model-name>`.
- `NHFLO/data`: reusable datasets and path helpers exposed through `nhflodata`.
- `NHFLO/tools`: reusable NHFLO-specific helpers.
- `gwmod/nlmod`: generic groundwater modeling, grids, layers, readers, plotting, xarray, and MODFLOW 6 helpers.
- `hydropandas`: reusable hydrological time-series functionality.

Search and place reusable functionality in this order: `nlmod`, `hydropandas`, `NHFLO/tools`, `NHFLO/data`, then model-specific code. If code is useful to a second model, extract it; if it is generic, upstream it.

## Data and privacy

- Read NHFLO data through `nhflodata`, preferably `nhflodata.get_paths.get_abs_data_path(...)`.
- Respect `NHFLODATA_LOCATION` for local unaltered or confidential data.
- Put reusable data in `NHFLO/data`; prefer GeoJSON in Dutch RD projection and CSV where practical.
- Keep data readers separated into path resolution, raw reading, cleaning, and grid interpolation. Reusable readers belong in `NHFLO/data`, `NHFLO/tools`, or `nlmod`.

## Python, environments, and validation

- Use `uv` only for Copilot work. Do not use Hatch except when explicitly maintaining legacy Hatch config or legacy CI.
- Keep Copilot environments separate from a user's `.venv`:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-copilot"
```

- Package repos (`NHFLO/data`, `NHFLO/tools`):

```powershell
uv sync --extra lintformat --extra test -q
uv run -q ruff format .
uv run -q ruff check --fix .
uv run -q pytest tests -v
```

- `NHFLO/models`: install the optional dependency extra with the same name as the modelscript folder:

```powershell
uv sync --extra lintformat --extra test --extra <model-name> -q
uv run -q ruff format modelscripts\<model-name>
uv run -q ruff check --fix modelscripts\<model-name>
uv run -q pytest --notebook-path <model-name> -v
```

Use the smallest targeted validation first. After `ruff check --fix`, rerun `ruff format` on changed paths. Run type checks such as `uv run -q mypy ...` only when configured. For documentation-only changes, review markdown and the focused diff instead of running unrelated tests. Do not claim validation unless it was run.

## `NHFLO/models` conventions

- Prefer one main modelscript that produces the complete requested results in one run.
- Keep scenarios in that script via explicit config, functions, or loops; split scripts only when model construction, data prep, or execution is substantially different.
- Adjacent `config.py`, `model.py`, `plot.py`, or `util.py` modules are fine when they keep the modelscript clear. Reusable code belongs in `NHFLO/tools`, `NHFLO/data`, or `nlmod`.
- Simple plot commands may be inline. Complex non-reusable analysis may be an adjacent postprocessing step. Reusable plotting or analysis belongs in `NHFLO/tools` or `nlmod`.
- Notebooks may support exploration or presentation, but model results must not depend on manual notebook execution. Commit cleared outputs only.
- Keep generated data, downloads, binaries, caches, and figures outside source-controlled modelscript folders.

## Git and PRs

- No AI signatures, generated-by lines, or co-author trailers.
- Keep PRs focused. If a change spans repositories, explain dependency direction and merge order.
