# NHFLO Copilot instructions

These instructions are the NHFLO organization baseline. Keep this file as `.github/copilot-instructions.md` in the `NHFLO/data`, `NHFLO/tools`, and `NHFLO/models` repositories. Always combine it with the repository-local README, installation notes, `pyproject.toml`, tests, and existing code conventions. More specific repository instructions take precedence.

## Working principles

- Inspect the repository before changing it. Do not assume that every NHFLO repository has the `NHFLO/models` layout.
- Keep changes small, reviewable, and scoped to one repository responsibility whenever possible.
- Prefer reusable implementations over model-local copies. If code is useful in more than one place, move it to the proper upstream package instead of duplicating it.
- Preserve scientific meaning: units, coordinate reference systems, sign conventions, layer numbering, time discretization, and MODFLOW package semantics must remain explicit.
- Do not commit secrets, confidential datasets, generated model output, downloaded binaries, caches, notebook outputs, or absolute local paths.

## Repository roles

- `NHFLO/models`: groundwater model applications, notebooks, run scripts, model-specific configuration, generated-model workflows, and model tests.
- `NHFLO/data`: reusable NHFLO datasets, package data, and path helpers exposed through `nhflodata`.
- `NHFLO/tools`: reusable NHFLO-specific helpers that are broader than one model but not generic enough for `nlmod` or `hydropandas`.
- `gwmod/nlmod`: generic groundwater-modeling functionality, including model grids, layers, MODFLOW 6 helpers, readers, xarray conventions, and plotting utilities.
- `hydropandas`: reusable hydrological time-series functionality.

When a change spans repositories, document the dependency direction and preferred merge order in the pull request.

## Where functionality belongs

Before adding helper code, search for existing functionality in this order:

1. `nlmod`
2. `hydropandas`
3. `NHFLO/tools`
4. `NHFLO/data`
5. Existing model-specific code in `NHFLO/models`

Place new functionality in the narrowest reusable location:

- Generic groundwater modeling, grid, layer, xarray, MODFLOW 6, data-reader, or plotting behavior belongs upstream in `nlmod`.
- Generic hydrological time-series behavior belongs in `hydropandas`.
- NHFLO-specific reusable helpers belong in `NHFLO/tools`.
- Reusable NHFLO datasets and data-location logic belong in `NHFLO/data`.
- Model-specific scenario setup, constants, local preprocessing, and one-off analysis belong in `NHFLO/models`.

If a helper is needed by a second model or repository, extract it rather than copying it.

## Data access and privacy

- Read reusable NHFLO data through the installed `nhflodata` package. Prefer `nhflodata.get_paths.get_abs_data_path(...)` over hard-coded paths.
- Respect the `NHFLODATA_LOCATION` environment variable for local unaltered or confidential datasets. Public `NHFLO/data` may contain altered data; code should still work with the configured local data location when it is set.
- Keep confidential data outside git. Do not embed user names, OneDrive paths, local drive roots, credentials, tokens, or network-share paths in source code or notebooks.
- If data is useful to more than one model or repository, add it to `NHFLO/data` instead of committing it under a model directory.
- Store reusable GIS data in GeoJSON in the Dutch RD projection when practical, and reusable tabular data in CSV when practical, so changes remain reviewable in git.
- Data-reading scripts should separate path resolution, raw-data reading, cleaning, and model-grid interpolation/resampling. Reusable readers belong in `NHFLO/data`, `NHFLO/tools`, or `nlmod` depending on scope.

## Local development and dependencies

- Follow the repository-local installation README first.
- Copilot must use `uv` for environment creation, dependency installation, command execution, and validation. Do not rely on Hatch for normal Copilot work. Existing Hatch configuration is legacy transition material; touch it only when the user explicitly asks to maintain or remove legacy Hatch configuration, or when a task directly targets an existing legacy CI workflow.
- NHFLO development commonly uses `uv` and editable installs. When working across repositories, the local checkout layout is typically:
  - `NHFLO/data`
  - `NHFLO/tools`
  - `NHFLO/models`
  - `gwmod/nlmod` on the development branch when contributing upstream
- For cross-repository development, editable installs are expected when the local repository exists:
  - `uv pip install -e "..\data"`
  - `uv pip install -e "..\tools"`
  - `uv pip install -e "..\..\gwmod\nlmod"`
- Do not add dependency managers, formatters, or test frameworks that the repository does not already use unless the task explicitly requires it.
- For Python code, use repository-configured Ruff settings and NumPy-style docstrings.

## If working in `NHFLO/models`

Each model belongs under `modelscripts/<model-name>`. Prefer one main modelscript that generates the complete requested model results in one run from the configured input data.

- Keep scenarios in the same modelscript through explicit scenario definitions, configuration dictionaries, functions, or loops.
- Create multiple modelscripts only when scenarios require substantially different model construction, data preparation, or execution workflows that would make one script unclear.
- Keep the main modelscript runnable from a clean environment with the configured data location. It should read input data, build the model, run the relevant scenarios, and write the expected outputs without hidden manual steps.
- To create an environment that can run a modelscript, install the optional dependency extra with the same name as the modelscript folder. For example, `modelscripts\14texel` requires the `14texel` extra.
- Use small adjacent modules such as `config.py`, `model.py`, or `util.py` only when they make the main modelscript clearer. Keep these modules model-specific; reusable code belongs in `NHFLO/tools`, `NHFLO/data`, or `nlmod`.
- A few simple plot commands may be part of the main modelscript. More complicated model-specific analysis or postprocessing that is not reusable elsewhere may live next to the modelscript as a postprocessing step. Analysis or plotting code that is useful for other scripts belongs in `NHFLO/tools` or `nlmod`.
- Notebooks may support exploration, documentation, or presentation workflows, but the model results should not depend on manually running notebook cells. Committed notebooks must have cleared outputs.

Keep generated model data, downloads, binaries, caches, and figures outside source-controlled modelscript directories. Use ignored model output, cache, and figure directories created by existing repository helpers, for example `nlmod.util.get_model_dirs(...)`, when available.

When adding a model, update the model dependencies, `uv` installation/run instructions, model-specific test and format workflows, README table, and pull-request checklist entries. Do not introduce new Hatch requirements for Copilot.

## Plotting guidance

- Use `nlmod.plot` utilities when they fit the task.
- Keep reusable plotting helpers in a `plot.py` module or an upstream package; keep one-off figure generation in the main modelscript, a model-specific postprocessing script, or a notebook.
- Plot scripts should save figures to generated output or figure directories, not to source-controlled code directories.
- Avoid duplicating plotting utilities between models. If plotting code is generic, upstream it to `nlmod`; if it is NHFLO-specific and reusable, move it to `NHFLO/tools`.

## Linting, formatting, testing, and validation

- Use `uv` only. Keep the Copilot environment separate from a user's existing `.venv` when installing or running tools.
- Start from the repository root and install only the extras needed for the task. For package repositories such as `NHFLO/data` and `NHFLO/tools`, use:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-copilot"
uv sync --extra lintformat --extra test -q
uv run -q ruff format .
uv run -q ruff check --fix .
uv run -q pytest tests -v
```

- For `NHFLO/models`, include the optional dependency extra with the same name as the changed modelscript folder instead of syncing all model environments:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-copilot"
uv sync --extra lintformat --extra test --extra <model-name> -q
uv run -q ruff format modelscripts\<model-name>
uv run -q ruff check --fix modelscripts\<model-name>
uv run -q pytest --notebook-path <model-name> -v
```

- Prefer the smallest targeted validation first, for example `uv run -q pytest tests\test_file.py -v` or `uv run -q pytest --notebook-path 14texel -v`; run broader tests when changing shared code, package APIs, data schemas, or cross-model behavior.
- Run Ruff formatting before Ruff linting. If `ruff check --fix` changes files, rerun `uv run -q ruff format ...` on the changed paths.
- Run repository-configured type checks when present, for example `uv run -q mypy ...`. Do not add a type checker or formatter just for Copilot.
- For documentation-only changes, review the readable markdown and inspect the focused git diff; do not run unrelated package or model tests.
- Keep notebooks executable and clear outputs before committing.
- Do not use Hatch as the validation or execution path unless the task explicitly targets legacy Hatch configuration.
- Do not claim validation passed unless it was actually run.

## Git and pull requests

- Do not add AI-assistant signatures, generated-by lines, or co-author trailers.
- Keep pull requests focused. Prefer one model, package, dataset, or upstreaming step per pull request.
- If a change depends on another repository, link the related pull request or commit and explain whether it must merge first.
- Pull requests should mention affected data sources, generated outputs, model assumptions, and validation commands when relevant.
