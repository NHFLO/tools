# Test plan — nhflotools, scoped to the 09pwnmodel2 closure

Implementation plan only; nothing here is implemented yet. Scope = the 13 nhflotools entry
points `models/modelscripts/09pwnmodel2/01_pwnmodel2.py` imports, plus their transitive
nhflotools closure (`pwnlayers.merge_layer_models`, `pwnlayers.utils`,
`panden.get_oppervlakte_pwn_shapes`). Target: a lean, meaningful pytest suite that runs
green on a fresh ubuntu GitHub runner with zero live network at test time.

Provenance: produced by a multi-agent review (9 module deep-reads, call-site trace,
git-history mining, existing-test audit, nlmod fixture survey, data/CI feasibility, two
competing designs, adversarial critique). All file:line claims below were verified against
HEAD during that review.

---

## 1. Current state (verified baseline)

- Existing suite: 18 tests, all passing on a healthy Python 3.14 env in <2 s
  (`test_postprocessing.py` 9: `interface_elevation` + one mocked
  `check_budget_discrepancy`; `test_well.py` 9: `get_wells_tata_dataframes` only).
- `test_panden.py` and `test_nhflo_utils.py` are 0 bytes; `test_hhnk.py` is a lone
  docstring — they collect nothing and misrepresent coverage.
- No `conftest.py` (a previous one plus `test_polder.py`/`test_lakes.py`/`test_pwnlayers.py`
  were deleted in b86583e; `__pycache__` remnants confirm).
- CI runs lint only (`.github/workflows/lint.yml`); there is no test workflow.
- `[tool.pytest.ini_options]` has only `testpaths`; pytest-env is installed but unused.
- The hatch-registered test env (`.direnv/test`) is a stale Python 3.11 env that can no
  longer resolve deps (`requires-python >=3.14`); it must be removed and recreated once.
- Both NHFLO/tools and NHFLO/data are public → CI needs no secrets. With
  `NHFLODATA_LOCATION` unset, `nhflodata.get_abs_data_path` resolves to mockup data
  packaged in the wheel; all datasets the script requests have mockup variants.
- **Blocker found: `pyarrow` is missing from dependencies** — `well.py:46`
  (`pd.read_feather`) needs it in production and in tests; `import pyarrow` fails in the
  tools venv. Add it to the **runtime** deps (not just test extra).

## 2. Untested-but-used surface

Zero coverage today for: `major_surface_waters` (both functions), `nhi_chloride`,
`panden` (both), `polder`, `well.get_wells_pwn_dataframe`,
`postprocessing.add_output_to_ds` / `plot_result_maps`, `pwnlayers.layers.get_top_from_ahn`,
all of `pwnlayers3` (layer model + plot). This includes the WEL/RIV/DRN/GHB/CHD boundary
conditions and the layer model — the physics core of 09pwnmodel2.

---

## 3. Architecture

### 3.1 `tests/util.py` (vendored helpers, ~90 lines)

nlmod's `tests/util.py` is not importable from an installed nlmod, so vendor equivalents:

| Helper | Builds |
|---|---|
| `make_rect_vertex_ds(nx=2, ny=2, delr=100.0, botms=(-10.0, -20.0), top=0.0, kh=5.0, transport=0)` | Hand-built rectangular pseudo-vertex ds: dims `(layer, icell2d)`, coords cell-center `x/y`, grid geometry `xv/yv/icvert`, vars `top/botm/kh/kv/area/idomain`, attrs `gridtype='vertex'`, `extent`, `model_name='test'`, `transport`. **No gridgen, no binaries, no network** — the single biggest feasibility lever. |
| `make_structured_ds(...)` | Thin wrapper over `nlmod.get_ds(extent, delr=100, ...)` with `download_exe=False` (param verified present in nlmod/dims/base.py:548). |
| `add_time(ds)` | `nlmod.time.set_ds_time(ds, time=[1], start='2022-01-01', steady=True)` — required before sim/tdis. |
| `make_gwf_disv(ds, tmp_path)` | `nlmod.sim.sim` + `tdis` + `nlmod.gwf.gwf` + `disv`; package construction only, never run. |
| `make_rws_gdf(rows)` | Synthetic EPSG:28992 GeoDataFrame factory, columns `OWMNAAM/peil/bweerstand/geometry`. |
| `write_mf6_listing(path, pct_disc, budgetkey)` | Emits a real MF6 listing **derived from the committed `.lst` of the smoke test's run** (§5.11), editing only the PERCENT DISCREPANCY numbers. Hand-rolled listing text risks `Mf6ListBudget.get_dataframes()` returning None → false-positive "Could not parse" tests. |

### 3.2 `tests/conftest.py`

| Fixture | Scope | Content |
|---|---|---|
| `_hygiene` (autouse) | function | Vendored verbatim from nlmod tests/conftest.py:39-46 (`plt.close('all')`, `gc.collect()`, clear xarray FILE_CACHE) + `monkeypatch.delenv('NHFLODATA_LOCATION', raising=False)` so a developer's data mount can never redirect mockup-resolution tests. |
| `vertex_ds` | function | Fresh `make_rect_vertex_ds()` per test — several targets mutate ds in place (`northsea`, `sfw_*`, `drn_*`, `thickness`); no shared mutable state. Build cost sub-millisecond. |
| `gwf_disv` | function | `(ds, sim, gwf)` in `tmp_path`. |
| `pwn_data_tree` | session | Synthetic bodemlagen tree: 7 boundary GeoJSONs (squares buffered +1e-3 m past cell edges to dodge the `min_area_fraction=1.0` float-equality trap), `botm/botm.geojson` point layer (all 14 W/S columns), `conductances/K*/KD*/C*` GeoJSONs, `triwaco_model_nhdz.geojson`, koppeltabel CSV. Immutable, written once. |
| `pwn_layer_model` | module (test_pwnlayers3_layers) | The one expensive result: `get_pwn_layer_model(...)` on a 4×4 synthetic vertex ds with `nlmod.read.regis.get_layer_names` monkeypatched. Read-only; the plot test takes a `.copy()`. |
| `chloride_nc` | session | Tiny `chloride_p50.nc`: 2 source layers (1-D `top/bottom` coords, power-of-two thicknesses), 3×3 grid at 250 m, **descending y** (matches the real file — ascending-y-only fixtures would pass while production fails). |

### 3.3 pyproject changes

- Add `pyarrow` to `[project] dependencies` (production requirement of `well.py:46`).
- `[tool.pytest.ini_options]`: `addopts = "--strict-markers"`,
  `markers = ["network: live web services, excluded in CI", "mf6: needs MODFLOW binaries"]`,
  `env = ["MPLBACKEND=Agg"]` (finally using the declared pytest-env),
  a named `filterwarnings` entry for the pyarrow feather FutureWarning.
- Fix the stale `tests/test_pwnlayers.py` reference in the lint script (pyproject.toml:85).
- Delete the empty shells `test_nhflo_utils.py` / `test_hhnk.py` (out of the 09pwnmodel2
  closure; they misrepresent coverage). Local housekeeping (not CI): `hatch env remove test`
  once; `tests/model_ws/` (51 MB untracked gridgen leftovers) can be deleted.

### 3.4 Mocking policy (named seams only)

Mock only at named network seams, with the reason stated in the test:
`nlmod.read.waterboard.download_data` (live HHNK ArcGIS REST, uncached at polder.py:40);
`nlmod.read.regis.get_layer_names` (OPeNDAP when called without ds);
`nlmod.plot.add_background_map` (contextily tile fetch);
`nlmod.read.rws.get_gdf_surface_water` (synthetic gdf substitution — one patch controls
`discretize_northsea`/`discretize_surface_water` too, verified major_surface_waters.py:42);
`nlmod.gwf.output.get_heads_da` / `nlmod.gwt.output.*` in the transport-derivation tests
(real loader path covered once by the MF6 smoke test). Everything else — `gdf_to_grid`,
`aggregate_vector_per_cell`, `build_spd`, `split_layers_ds`, `discretize_*`,
`calculate_thickness`, `get_isosurface` — runs the **real nlmod code**, so the suite doubles
as an nlmod-@dev compatibility canary (that canary already caught the
`get_isosurface(left=...)` staleness failure).

### 3.5 Assertion policy

Exact (`assert_array_equal` / `==`) by default; geometry axis-aligned on grid-multiple
coordinates so shapely areas are exact floats; expected values derived in-test from first
principles, never pasted from implementation output. The only tolerances, each named:
IMS solver `dvclose` (MF6 smoke, atol=1e-8), `rtol=1e-12` on area-weighted means (pandas
groupby accumulation order vs the test's reconstruction), `rtol=1e-12` on linear-griddata
plane reproduction (Qhull barycentric arithmetic).

**Baseline-validation protocol** (global CLAUDE.md doctrine): every test tagged
`[regression: <commit>]` must be run against the pre-fix baseline
(`git checkout <commit>^ -- src/nhflotools/<file>` in a scratch worktree) and confirmed to
FAIL there, then confirmed to pass on HEAD, before it is trusted.

---

## 4. Test files and per-test specs

### 4.1 `tests/test_major_surface_waters.py` (new)

1. **test_get_chd_ghb_data_known_answer_cond_area** — 2×2 vertex ds (100 m cells);
   synthetic gdf: sea polygon (`OWMNAAM='Hollandse kust (kustwater)'`) over cell 0,
   `'IJsselmeer'` (peil=-0.2, bweerstand=2.0) exactly covering cell 3. Assert
   `rws_oppwater_cond[3] == 10000/2.0`, `area[3] == 10000.0`, `stage[3] == -0.2`, dry
   cells `== 0.0` (pins the zeros-not-NaN contract downstream code relies on). Catches the
   conductance-formula bug class (the panden 83365c7 1e4× error, identical code shape).
2. **test_stage_override_only_where_ijsselmeer_intersects** — Noordzeekanaal wins cond in
   cell 2; IJsselmeer (peil=p) also intersects cell 2 but not cell 3. Assert
   `stage[2] == p` and `stage[3]` bit-identical to winner-take-all (the
   `xr.where(da_peil.isnull())` identity). Catches override-mask inversion.
3. **test_input_ds_mutation_northsea_and_extrapolation** — plant an all-NaN `botm/kh`
   column under the sea cell. Afterwards: input ds has `northsea == 1` exactly there, and
   the column equals its Euclidean-nearest valid neighbor exactly. Catches the
   discarded-`extrapolate_ds`-return hazard (in-place mutation contract, nlmod
   dims/base.py:286-301): a copy-semantics change in nlmod leaves NaNs under the sea and
   this fails loudly.
4. **test_chd_ghb_float_branch_masks_disjoint_layers** — `chd_ghb_from_major_surface_waters`
   on `vertex_ds` + `gwf_disv`; cells: sea / wet / two dry; one wet cell `idomain[0]=0`;
   **plus one sea cell `idomain[0]=0` among active sea cells** (must be absent from CHD —
   the partial-absence case). `sea_stage=0.04`. Assert: `sfw_stage` NaN exactly where
   `northsea==1`, `== rws_oppwater_stage` elsewhere; `sfw_cond == 0.0` on sea; **GHB ∩ CHD
   cell sets = ∅** (boundary-type disjointness — a cell that is both GHB and fixed-head is
   non-physical); every CHD rec head `== 0.04`, aux `== 18000.0` (`SEA_CHLORIDE_MG_L`);
   `ts_sea is None`; the idomain-blocked GHB row sits in layer 1 (first active), the
   blocked CHD cell absent (layer-0-only contract). One setup, six bug classes.
5. **test_chd_ts_branch_wiring** — `sea_stage=[(0.0, 0.04), (365.0, 0.04)]`. Assert every
   CHD rec head is the **literal string** `'sea_stage'`; `chd.ts` has
   `time_series_namerecord == 'sea_stage'`, linear interpolation, timeseries equal to the
   input list (a flat series is semantically the constant branch).
   `[regression: b42fec1]` — a list-valued sea_stage never worked (NameError + wrong
   namerecord).
6. **test_chd_none_guard_returns_three_tuple** — all sea cells `idomain[0]=0`, list-branch
   call. Assert `(ghb, None, None)`, no raise. `[regression: 9d6c40c]`
   (AttributeError on `chd.ts.initialize`).
7. **test_gdf_permutation_invariance** (lowest priority in file) — permute gdf rows; all
   three output vars bit-identical (winner-take-all must be order-free; no area ties by
   construction). Catches a last-wins loop rewrite. Cheap.

*Flagged, deliberately not tested (would entrench defects):* the
`isinstance(sea_stage, float)` int-rejection quirk; GHB silently zeroed where sea and
canal overlap. → §8 issues.

### 4.2 `tests/test_nhi_chloride.py` (new)

Uses `chloride_nc`; called without `cachedir` (decorator short-circuits) except test 7.

1. **test_constant_field_identity** — source ≡ 1000 covering the full vertical extent,
   no sea → output == 1000.0 everywhere, layers preserved, `units == 'mg/l'`. Any
   non-convex aggregation fails. Exact.
2. **test_weighted_mean_known_answer** (parametrized ×2) — (a) model layer `[0,-8]`, voxels
   c=100/300 → exactly `(4*100+4*300)/8 = 200.0`; (b) layer `[-2,-6]` clipping 2 m of each
   voxel → `200.0` with different weights; a `[0,-4]`-only layer → `100.0`. Catches
   clip/weighting off-by-ones. Exact (integer-valued arithmetic).
3. **test_nan_voxel_excluded_from_denominator** — one voxel NaN → column output `== c1`
   exactly, not diluted toward 0. Catches the skipna-numerator/keep-denominator bug (the
   exact bug that *does* exist at postprocessing.py:182 — see §8).
4. **test_sea_override_layer0_only** — sea cell: layer 0 `== 18000.0` exactly; deeper
   layer unchanged. Catches positional `da.values[0]` misalignment.
5. **test_deep_layer_bfill_ffill** — model layers wholly below and wholly above the source
   stack → output NaN-free; deep layer equals last covered value, top layer equals first
   covered value (asymmetric column pins bfill-before-ffill order).
   `[regression: 83365c7]` — all-NaN deep layers poisoned the transport IC.
   **Contingency:** the recon flags a possible empty-valid-points crash in the fillnan
   step *before* the bfill/ffill rescue (nhi_chloride.py:44-52 ordering). If this test
   crashes rather than fails, that is a live bug find — report it, don't reshape the
   fixture to avoid it.
6. **test_vertex_grid_path** — 2×2 pseudo-vertex ds (production grid type): output dims
   `(layer, icell2d)`; one cell outside the source's horizontal extent → NaN → vertex
   `fillnan` path fills it with its nearest neighbor's value exactly. Covers the
   production-only dispatch that structured tests never touch (a constant field would
   leave nothing to fill and make this a near-no-op).
7. **test_cache_roundtrip_identical** — attrs `gridtype/extent`,
   `cachedir=tmp_path, cachename='chloride'`: `chloride.nc` + `.pklz` written; second call
   bit-identical values. Pins the production cachedir path (call site always passes it).

### 4.3 `tests/test_panden.py` (fill the 0-byte file)

Real tiny shapefile via `gpd.to_file` (exercises the real read path); structured ds must
include `kh` (build_spd reads it unconditionally).

1. **test_get_oppervlakte_pwn_shapes_name_filter** — Naam
   `['ICAS-noord','IKIEF-3','VIJVER']`: stages exactly `{2.8, 5.8}`, `c == 1.0`,
   `rbot == stage - 2.0` elementwise, VIJVER row absent, caplog warning.
   `[regression: e16b418 (#41)]` — VIJVER used to abort the model run.
2. **test_riv_known_answer_single_cell** — 50×50 ICAS square inside one cell → exactly one
   spd record `[(0,iy,ix), 2.8, 2500.0, 0.8, 0.0, 'ICAS…']` (aux before boundname).
   `[regression: 83365c7]` — baseline aggregated resistance `c` instead of `area/c`; cond
   would be `1.0`, a silent ~1e4× physics error.
3. **test_riv_cond_conservation_across_cells** — one rectangle straddling two cells (areas
   2500/7500, axis-aligned → exact). Per-cell cond == intersected area and
   `sum(cond) == 10000.0` exactly (partition conservation). Same baseline, different
   failure mode (per-piece vs per-cell aggregation).
4. **test_riv_mixed_names_cell** — ICAS + IKIEF sharing one cell. Stage
   `== (2.8*a1+5.8*a2)/(a1+a2)` (rtol=1e-12, named: groupby accumulation order),
   `rbot == 0.8` (min), `cond == a1+a2`; assert build_spd's stage-clip warning did not
   fire. Characterizes the mixed-cell aggregation mapping.
5. **test_riv_lay_of_rbot_placement** — 2-layer ds `botm=[1.0, -10]`, rbot=0.8 → record in
   layer **1**, not 0. Catches the first-crossing off-by-one.
6. **test_riv_empty_intersection_returns_none** — polygons outside extent → `None`,
   `ds.attrs['ssm_sources']` untouched. `[regression: 232de8d]`
7. **test_riv_ssm_registration_idempotent** — `transport=1`, call twice → `'riv'` appears
   exactly once; `transport=0` → attrs untouched. `[regression: c613863 (#39)]`
8. **test_riv_pregridded_frame_raises** — feed a frame with duplicate index / already
   gridded (the #114 generalization): `gdf_to_grid` raises ValueError
   (nlmod grid.py:1996-1997). Negative test documenting the single-intersection contract
   so the models-repo crash class is pinned on the tools side.

### 4.4 `tests/test_polder.py` (recreate deleted file)

All tests monkeypatch `nlmod.read.waterboard.download_data` (named: live HHNK ArcGIS REST,
uncached at polder.py:40). Grid = `vertex_ds` (+ float `top`, `ahn`, `northsea`) + `gwf_disv`.

1. **test_drn_known_answer_and_cbot_scaling** (parametrized cbot ∈ {1.0, 2.0}) — polygon
   fully covering cell A, `summer==winter==s`: `drn_cond[A] == area/cbot` exactly (1/cbot
   proportionality across params), `drn_elev[A] == s`.
2. **test_drn_elev_nan_skipping_mean** — stages (1.0, 3.0) → 2.0; (1.0, NaN) → 1.0,
   exactly (one-sided nanmean, not griddata).
3. **test_drn_fallback_partition_and_layer** — 4 cells: covered / uncovered land
   (`ahn=4.25`) / `northsea==1` / uncovered land with `idomain[0]=0`. Uncovered land gets
   `elev == ahn` and `cond == area/cbot` (**nonzero** — the historical silent failure);
   sea cell `drn_cond` NaN and **absent from the DRN reclist**; blocked cell's record in
   layer 1. `[regression: 83365c7]` — zero-init gave uncovered land zero-conductance
   drains and drained the sea.
4. **test_drn_empty_download_returns_none_ds_untouched** — empty GeoDataFrame → `None`;
   `drn_elev`/`drn_cond` **not** in ds. `[regression: a048d98]`
5. **test_drn_duplicate_index_dedup** — download index `['A','A','B']` → no ValueError
   from `gdf_to_grid`; assert dedup via the area-weighted **stage** of the covered cell
   plus record presence. (Do **not** assert cond from intersected areas: verified
   polder.py:59 computes `cond = full-cell area / cbot`, so an intersected-area cond
   assertion would fail against correct code.)
6. **test_drn_both_stages_nan_nearest_fill** — polygon in cell A with both stages NaN,
   valid stage s in cell B → `drn_elev[A] == s` exactly (nearest donor, member of the
   valid set). Documents the all-NaN crash boundary without pinning scipy's error.

*Flagged, not tested:* full-cell (not intersected-area) conductance for sliver overlaps —
design decision to raise upstream, not entrench. → §8.

### 4.5 `tests/test_well.py` (extend; keep the existing 9 tata tests, add none for tata)

Real tiny files: geojson via `gpd.to_file` (deliberately exercises the GeoJSON round-trip
that stringifies `sec_nput` — the reason for the coercion at well.py:59), feather via
`to_feather` (needs pyarrow, §1 blocker).

1. **test_pwn_q_known_answer_conservation** — 3 wells tag T1, `sec_nput=3` (one given as
   string `"3"`), feather `T1 = [-30,-30,-30]` m³/h; second tag with odd asymmetric series
   `[-10,-20,-60]` on one well (`sec_nput=1`); an `'ophaal tijdstip'` datetime column.
   Assert each T1 well `Q == -240.0` exactly, `sum == -720.0 == 24*median` (the WEL
   mass-balance contract the MAW branch scales back up); odd-series well
   `Q == 24*(-20.0)`; the datetime column never contaminates the median; `rw == 0.25`,
   `CONCENTRATION == 0.0`. `[regression: 83365c7]` (`numeric_only` + factor-24/nput).
2. **test_pwn_drops_bad_rows_warns_keeps_infiltration** — rows: unmapped tag, `sec_nput=0`
   (inf Q), zero-median tag, and one **positive**-median infiltration well. Assert exactly
   the three bad rows dropped, warning count == dropped count, survivors finite and
   nonzero, the positive-Q well survives with positive sign (drop mask is sign-symmetric).
   `[regression: 83365c7 + c613863]`
3. **test_pwn_flow_product_error_contract** (parametrized) — `'timeseries'` →
   `NotImplementedError`, `'bogus'` → `ValueError` (documents that the feather read
   precedes dispatch).

*(Design A's tata deepest-layer proposal was cut: `well.py:143-146` shows the `lay>0` path
is identical for mid and deepest layers — no new branch.)*

### 4.6 `tests/test_postprocessing.py` (extend; keep the existing 9)

1. **test_check_budget_real_listing** (parametrized ×3) — listing text from
   `write_mf6_listing` (provenance §3.1), parsed by **real** `flopy.utils.Mf6ListBudget`:
   `|disc| = 0.99` passes; `= 1.0` raises (inclusive `>=` boundary); garbage file →
   `RuntimeError` "Could not parse". Catches budgetkey-string and flopy API drift the
   existing fully-mocked test cannot.
2. **test_add_output_freshwater_head_identity** — monkeypatch the three nlmod output
   loaders to synthetic `(time=2, layer=3, icell2d=3)` arrays; conc ≡ 0, `drhodc` attr
   set. Assert `freshwater_head == head_filled` **exactly** (ρ = ρ_ref identity). Catches
   density-correction wiring (z-term sign, drhodc pickup).
3. **test_add_output_constant_conc_identities** — conc ≡ 3000 (between thresholds):
   `concentration_mean == 3000.0` exactly regardless of unequal layer thicknesses;
   `dconcentration_mean.isel(time=0) == 0` identically; `grensvlak_zoet == ds['top']` and
   `grensvlak_brak == botm[-1]` per the pinning rules; one planted head NaN →
   `head_filled == head` wherever head is finite (bfill conservation).
4. **test_add_output_zoet_brak_ordering** (graft from Design A) — monotone-with-depth
   synthetic conc: `grensvlak_zoet >= grensvlak_brak` everywhere (the fresh interface is
   never deeper than the brackish one).
5. **test_interface_bounds_random_profiles** (graft from Design A) — property test,
   `np.random.default_rng(0)`, ~50 random profiles:
   `botm[-1] <= interface_elevation(...) <= top` per column — bounds currently unasserted
   for interpolated columns. Exact bound check.
6. **test_plot_result_maps_filenames_and_early_return** (parametrized) — monkeypatch
   `nlmod.plot.add_background_map` (named: contextily tile fetch, the module's only
   network call); tiny vertex ds with `freshwater_head/grensvlak_*` (with `threshold`
   attrs) + ctop, nper=2, `iper=-1`. Assert figdir contains **exactly**
   `{doorsnedelijnen.png, map_head_L0_t1.png, map_conc_L0_t1.png, grensvlak_zoet_t1.png,
   grensvlak_brak_t1.png}` (t1 = the iper-normalization known answer). Variant (b): ds
   without `freshwater_head`, `ctop=None` → only `doorsnedelijnen.png`, no raise. Variant
   (c): ds with `drn_elev` (the call site always has it — polder.py:81-82 sets it; the
   branch is live, contrary to the recon's first read) → `oppervlaktewater.png` in the set.

### 4.7 `tests/test_pwnlayers_get_top.py` (new — `get_top_from_ahn`)

1. **test_pure_griddata_nearest_known_answer** — anisotropic 5-cell layout where the
   x-nearest and y-nearest donors differ; **query strictly off the q1==q2 diagonal and
   donors off p1==p2** (a query at the origin is provably invariant under a one-sided
   (y,x)↔(x,y) swap — the geometry must make the swap detectable); flags off (touches
   zero nlmod code). Assert the NaN cell receives its true Euclidean-nearest value and all
   valid cells are bit-identical. Catches a silent axis-order swap during refactoring.
2. **test_no_nan_identity** — NaN-free ahn → output identical, no raise (pins the
   empty-qpoints scipy edge, scipy-version-sensitive).
3. **test_fill_priority_peil_constant_partial** — monkeypatch the `nlmod.read.rws` trio to
   hand-built Datasets. Three NaN cells: fully-covered water that is also in the northsea
   mask → gets **peil** (priority order); sea-only cell → gets `0.0` with
   `replace_northsea_with_constant=0.0` (**the falsy-constant guard** — production passes
   0.0; a refactor of the `is not None` test to truthiness silently disables the sea fill
   and this fails); partially-covered cell (area < cell area) → falls through to nearest
   interpolation (the isclose full-coverage gate). Highest-value test in the file.
4. **test_missing_ahn_raises_valueerror** — one-line error-surface pin (guards silent key
   rename).

### 4.8 `tests/test_pwnlayers3_layers.py` (new — highest-risk module)

Pure seams first, then the merge, then one offline integration.

1. **test_fix_missings_botms_both_copies** (parametrized over the two divergent duplicates
   `pwnlayers3.layers` and `pwnlayers.utils` — locks them together; named maintenance
   trap) — 3-layer × 4-cell botm, mid-column NaN, one botm crossing above the layer above;
   expectation hand-computed from ffill + `minimum.accumulate`. Assert exact equality,
   NaN-free, monotone non-increasing, `<= top`, **idempotent** (`f(f(x)) == f(x)`),
   **input object unmodified** (purity — `[regression: df20a42]`, the caller assumed
   in-place mutation and discarded the fix), ValueError on NaN top.
2. **test_get_thickness_telescoping** — `thickness[k] == botm[k-1] - botm[k]` exact,
   `sum('layer') == botm[0] - botm[-1]`, labels are the lower layers, W11 absent. Catches
   the sign/labeling class `[regression: 479d673]`.
3. **test_guard_zero_thickness** — thickness `[0.0, 1e-12, 0.5]` → fill_value exactly at
   the isclose-zero entries, others untouched; identity when no zeros.
4. **test_kh_kv_harmonic_and_anisotropy** (graft from Design A — the physics core:
   resistances → conductivities) — 2×2 vertex ds + `pwn_data_tree` variant with constant-c
   conductance polygons. Assert: single constant-c polygon fully covering the grid →
   `kv == d/c` and `kh == d·anisotropy/c` exactly; two half-covering polygons c1=2, c2=4 →
   `inv_c == 0.5·(1/2 + 1/4) == 0.375` exactly; W layers `kv == kh/anisotropy` exactly on
   the mask; S layers on the c-path `kh/kv == anisotropy` exactly where thickness > 0.
5. **test_kh_nhdz_branch_known_answer** (critique gap) — a cell inside the
   `triwaco_model_nhdz` region: `kh == KD/d` from the point-griddata path
   (layers.py:857-875); also covers the empty-Bergen-region 0-cell GridIntersect hazard
   (layers.py:887-889) by keeping one region empty and asserting no crash and no
   contamination.
6. **test_interpolate_da_nearest_linear_and_view_mutation** — 4-cell line;
   `_interpolate_da` middle cell missing → nearest donor value; linear midpoint == mean of
   the two valid values (representable); `ismissing` empty → strict no-op; **assert the
   parent Dataset's array actually changed** (the `.loc`-on-`sel`-view contract — an
   xarray copy-semantics change turns transition interpolation into a silent no-op).
7. **test_combine_identity_reduction** — synthetic REGIS (NaN-free) + OTHER with all-False
   mask/transition, pure-1:1 koppeltabel → output `kh/kv/botm` bit-identical to REGIS,
   categories all 1. Baseline sanity for the whole merge.
8. **test_combine_routing_and_split_conservation** — koppeltabel with 1:1, 1:2 (REGIS
   split), 2:1 (OTHER split), one NaN-uncoupled deep row; mask True on half the cells.
   Assert: category-2 cells carry OTHER values exactly, category-1 REGIS exactly; per
   cell, sum of split-sublayer thicknesses == original layer thickness (exact — same-float
   subtraction chains); group bottoms preserved; uncoupled layers category 1. Catches the
   positional koppeltabel/split-alignment hazard — the scariest silent-corruption vector
   in the codebase.
9. **test_combine_transition_convexity** — transition band between mask and REGIS regions:
   interpolated transition kh/botm lie within `[min, max]` of surrounding category≠3
   values **and differ from their pre-merge REGIS values** (proves interpolation actually
   ran; complements test 6).
10. **test_ratios_forward_inverse** — `_compute_thickness_ratios`: ratios sum to 1 per
    cell exactly, reproduce actual fractions, `1/N` at zero-thickness;
    `_apply_ratios_to_botm` with equal ratios reproduces `split_layers_ds`'s equal-split
    botms exactly (forward+inverse identity). Catches the group-top `first_idx-1`
    off-by-one.
11. **test_get_pwn_layer_model_offline_integration** (module fixture `pwn_layer_model`) —
    4×4 synthetic vertex ds_regis (3 fake REGIS layers + `'mv'`),
    `nlmod.read.regis.get_layer_names` monkeypatched (named: OPeNDAP in the first line),
    `pwn_data_tree` with one conductance polygon deliberately `VALUE=0`. Assert
    postconditions: `kh/kv/botm` NaN-free; **`kh, kv > 0` everywhere** (the zero/inf guard
    → NaN → fill path; `[regression: df20a42/5daf902]` zero-kh NPF crash); botm monotone
    non-increasing; output `top` identical to input; W-layer `kv == kh/10` exactly on
    masked cells; diagnostics `cat_botm/botm_method/kh_method/kv_method` present **with
    `flag_values`/`flag_meanings` attrs** (the plot module's hard requirement); botm
    source points on an affine plane z=ax+by+c reproduced at interior cell centres
    (rtol=1e-12, named: Qhull barycentric arithmetic). Parametrized negatives: NaN in top →
    ValueError; ds_regis missing a layer → ValueError. Budget ≈ 1–3 s.
12. **test_area_passthrough_values** — ds_regis with an `'area'` var → output area values
    identical; without → computed. `[regression: d62ebbf + 83365c7]` (eager-default
    KeyError). **Values-passthrough only** — do NOT assert `get_area` is not called:
    verified layers.py:232 `ds_regis.get("area", nlmod.dims.get_area(ds_regis))` evaluates
    the default eagerly even when `area` exists, so a spy assertion fails on HEAD. The
    eager evaluation is filed as a fix candidate instead (§8).

### 4.9 `tests/test_pwnlayers3_plot.py` (new)

1. **test_parse_flag_labels_known_answers** (parametrized) — the three **literal**
   `flag_meanings` strings copied from layers.py into the test as the independent source →
   exact label lists of lengths 5/7/4, each matching its `flag_values` length (the gate at
   plot.py:243 silently drops all tick labels on mismatch); adversarial cases: trailing
   `;`, entry without `N:` prefix, text after `(` truncated, underscores→spaces. Pure,
   instant.
2. **test_load_project_and_overlay** — tmp `botm/botm.geojson` (EPSG:28992), points at
   perpendicular distances 30 m and 80 m from line `[(0,0),(100,0)]`,
   `buffer_distance=50`, one point at exactly 50 (boundary `<=`); shuffled non-RangeIndex
   and non-layer column order. Assert exact `d_along` projections, far-point exclusion,
   boundary inclusion, returned keys == layer_names ∩ columns **in layer_names order**
   (positional-alignment hazard). Then feed the dict (plus one all-NaN layer and one z
   outside `[zmin, zmax]`) to `_overlay_source_botm` on an Agg Axes: number of
   PathCollections == layers with ≥1 valid point; no NaN/out-of-range offsets.
3. **test_plot_diagnostic_cross_sections_end_to_end** — reuse `pwn_layer_model` (`.copy()`)
   + ds_regis with `xv/yv/icvert` injected (replicating the call-site mutation), midline
   through the synthetic extent, `data_path_2024=pwn_data_tree` (the call site always
   passes it — 01_pwnmodel2.py:604). Assert `(fig, axes)` with `axes.size == 6` and cat
   colorbar ticklabels exactly `['REGIS', 'PWN', 'Transition']`; close fig. Justified
   against the no-smoke rule: the named KeyError regressions (missing flag attrs,
   `layer_pwn` rename, `return_diagnostics=False` ds) all make precisely this call raise.

### 4.10 `tests/test_nhflodata_contract.py` (new)

1. **test_mockup_paths_and_koppeltabel_columns** (parametrized ~10 cases) — with
   `NHFLODATA_LOCATION` deleted, `get_abs_data_path(name, 'latest')` + the hardcoded
   relative file must exist for every (dataset, file) pair nhflotools reads:
   `Panden_ICAS_IKIEF.shp`, `pumping_infiltration_wells.geojson`, `sec_flows.feather`,
   both tata geojsons, `chloride_p50.nc`, `bodemlagenvertaaltabelv2.csv` (also read the
   20 KB and assert columns `'Regis II v2.2'` / `'ASSUMPTION1'`), `botm/botm.geojson`,
   `boundaries/S11/S11.geojson`, `boundaries/triwaco_model_nhdz.geojson`. Since
   `get_abs_data_path` only **warns** on missing paths (get_paths.py:109-110), this is the
   sole automated guard against a data-repo restructure silently breaking 09pwnmodel2.
   Instant (stat calls; the 31 MB nc is never opened).

### 4.11 `tests/test_mf6_smoke.py` (new, marker `mf6`)

1. **test_tiny_run_budget_and_output_loading** — nlmod test_015 recipe: 3×2 cells × 5
   layers structured ds, CHD=1.0 on the edge mask, `write_and_run`. Then, on the **real**
   files: `check_budget_discrepancy(ws, name, transport=False)` passes (an all-CHD steady
   model closes its budget); `add_output_to_ds(ds, ws, name, transport=False)` returns
   `(ds, None)` with `ds['head'] == 1.0` everywhere (atol=1e-8 — named: IMS iterative
   solver converged to dvclose, not algebraic exactness). The one end-to-end pipeline test
   with a physical invariant, and the only coverage of the real `.hds`/`.grb`/listing
   loader paths the monkeypatched tests bypass. Its committed `.lst` (~2 KB) is the
   provenance source for `write_mf6_listing` (§3.1). ~3–8 s.

---

## 5. Coverage map (every 09pwnmodel2-used function)

| Function | Disposition |
|---|---|
| `pwnlayers.layers.get_top_from_ahn` | §4.7 (4 tests) |
| `pwnlayers3.layers.get_pwn_layer_model` (+ get_ds/get_botm/get_kh/get_kv, combine, fix_missings ×2) | §4.8 (12 tests) |
| `major_surface_waters.get_chd_ghb_data_from_major_surface_waters` | §4.1.1–3, 7 |
| `major_surface_waters.chd_ghb_from_major_surface_waters` | §4.1.4–6 |
| `nhi_chloride.get_nhi_chloride_concentration` | §4.2 (7 tests) |
| `well.get_wells_pwn_dataframe` | §4.5 (3 tests) |
| `well.get_wells_tata_dataframes` | existing 9 tests; **no additions** (nearest-cell, kd threshold strictness, chloride-warning boundary, screen offsets, IndexError contract already covered; more violates leanness) |
| `polder.drn_from_waterboard_data` | §4.4 (6 tests) |
| `panden.riv_from_oppervlakte_pwn` + `get_oppervlakte_pwn_shapes` | §4.3 (8 tests) |
| `pwnlayers3.plot.plot_diagnostic_cross_sections` (+ helpers) | §4.9 (3 tests) |
| `postprocessing.check_budget_discrepancy` | existing + §4.6.1 + §4.11 real run |
| `postprocessing.add_output_to_ds` | §4.6.2–5 + §4.11 real run |
| `postprocessing.plot_result_maps` | §4.6.6 |
| `postprocessing.interface_elevation` | existing 8 tests + §4.6.5 bounds property |

**Named exclusions:**

- **Full 09pwnmodel2 pipeline / transport run** — needs live REGIS, AHN5, HHNK services
  plus a minutes-long MF6 flow+transport run; not CI-realistic. The pipeline obligation is
  discharged by §4.11 (real run + budget closure) and §4.8.11 (offline layer-model
  integration). A future opt-in `network`-marked nightly is noted, not planned.
- **`pwnlayers3.layers.get_top`** — dead at the call site (script imports
  `get_top_from_ahn`); default path hits `download_bathymetry` (network).
- **Legacy `pwnlayers.get_bergen_botm` / `get_mensink_botm`** history candidates (883e517,
  479d673) — outside the 09pwnmodel2 closure; the sign/labeling bug class is covered by
  §4.8.2's analog in pwnlayers3.
- **numba-accelerated `get_isosurface` path** — CI env has no numba, so only the numpy
  fallback is exercised; numba/numpy divergence is untestable there. Stated, not hidden.
- **`get_transition` monotonicity** — acceptable drop; get_ds's internal validation
  (layers.py:400-417) executes inside §4.8.11.
- **`recharge_utils`** — lives in the models repo, zero nhflotools calls, has its own test
  file there; port only if it migrates to tools.
- **hhnk.py, nhflo_utils.py, bofek.py, geoconverter, bergen/berging utils** — not in the
  09pwnmodel2 closure; out of scope by the task's restriction.

---

## 6. CI workflow

New `.github/workflows/test.yml` beside lint.yml:

```yaml
name: test
on:
  push: {branches: [main]}
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    env:
      MPLBACKEND: Agg
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version-file: pyproject.toml}   # 3.14, same as lint.yml
      - name: Resolve moving deps for cache key
        id: deps
        run: |
          echo "data_sha=$(git ls-remote https://github.com/NHFLO/data.git main | cut -f1)" >> "$GITHUB_OUTPUT"
          echo "nlmod_sha=$(git ls-remote https://github.com/gwmod/nlmod.git dev | cut -f1)" >> "$GITHUB_OUTPUT"
      - uses: actions/cache@v4          # uv download cache incl. the ~235 MB nhflodata build
        with:
          path: ~/.cache/uv
          key: uv-${{ hashFiles('pyproject.toml') }}-${{ steps.deps.outputs.data_sha }}-${{ steps.deps.outputs.nlmod_sha }}
      - run: pipx install hatch
      - name: Sync test env
        run: hatch env create test      # uv installer per [tool.hatch.envs.default]
      - uses: actions/cache@v4          # MODFLOW binaries + flopy exe metadata
        id: mfcache
        with:
          path: |
            ~/mfbin
            ~/.local/share/flopy
          key: mfbin-${{ steps.deps.outputs.nlmod_sha }}
      - name: Download MODFLOW binaries
        if: steps.mfcache.outputs.cache-hit != 'true'
        run: hatch run test:python -c "import os, nlmod; nlmod.util.download_mfbinaries(bindir=os.path.expanduser('~/mfbin'))"
      - name: Run tests
        run: hatch run test:test -- -m "not network" --durations=15
```

Notes (each verified during review):

- `~/.local/share/flopy` (the `get_modflow.json` exe metadata flopy writes outside the
  bindir) must be cached alongside `~/mfbin`, otherwise a cache hit skips the download and
  exe resolution fails. `bindir` needs `os.path.expanduser` — a literal `'~/mfbin'` is not
  tilde-expanded by Python. (Confirm the metadata path on the runner at implementation
  time; it is platform-dependent.)
- `NHFLODATA_LOCATION` deliberately unset (and force-deleted in conftest) → mockup
  resolution inside the installed wheel is what is exercised.
- The `git ls-remote` SHAs make the unpinned `nlmod@dev` / `nhflodata@main` deps
  cache-correct (new upstream commit → fresh env) while caching between pushes. The
  observed stale-env failure class (nlmod 0.9.1dev lacking `get_isosurface(left=...)`) is
  exactly what this prevents. Longer-term: pin both to SHAs in pyproject.
- Both repos are public — no tokens needed.
- Budget: pytest total ≤ 60 s (estimate ~35 s); job total ≤ 6 min cold, ≤ 2.5 min warm.

---

## 7. Implementation order (value-ranked from the regression analysis)

1. `test_panden.py` — silent 1e4× physics error class, empty file today
   (baselines 83365c7/e16b418/232de8d/c613863)
2. `test_polder.py` — silent zero-cond drains + sea drains (83365c7/a048d98)
3. `test_well.py` PWN additions — silent mass-balance factor (83365c7/c613863);
   **prerequisite: add pyarrow to runtime deps**
4. `test_major_surface_waters.py` — crash class + boundary-condition correctness
   (9d6c40c/b42fec1)
5. `test_nhi_chloride.py` — transport-IC poisoning (83365c7)
6. `test_pwnlayers3_layers.py` — deepest silent-corruption surface (df20a42/5daf902/479d673)
7. `tests/util.py` + `conftest.py` + CI workflow — prerequisite for 1–6, build alongside 1
8. `test_pwnlayers_get_top.py`, `test_postprocessing.py` additions, `test_pwnlayers3_plot.py`
9. `test_nhflodata_contract.py`, `test_mf6_smoke.py` (smoke test also generates the
   committed `.lst` fixture used by §4.6.1)

For every `[regression: <commit>]` test: before merging, restore the pre-fix file
(`git checkout <commit>^ -- src/nhflotools/<file>` in a scratch worktree), confirm the
test FAILS, revert, confirm it passes on HEAD.

## 8. Issues found during review (NOT tests — tests must not entrench them)

All verified against HEAD and filed on GitHub.

| # | Finding | Status |
|---|---|---|
| 1 | `pd.read_feather` needs pyarrow, not a declared dependency (well.py:46) | **Fixed in this PR** — added to runtime deps |
| 2 | `concentration_mean` NaN/thickness-denominator bias (postprocessing.py:182) | [#60](https://github.com/NHFLO/tools/issues/60) |
| 3 | Eager `get_area` default at pwnlayers3/layers.py:232 | [#61](https://github.com/NHFLO/tools/issues/61) |
| 4 | `isinstance(sea_stage, float)` mishandles an int stage (major_surface_waters.py:102) | [#62](https://github.com/NHFLO/tools/issues/62) |
| 5 | Polder conductance uses full-cell, not intersected, area (polder.py:59) | Already filed as [#51](https://github.com/NHFLO/tools/issues/51) |
| 6 | Sea override precedes horizontal fill, seeding coastal land with 18000 mg/l | [#63](https://github.com/NHFLO/tools/issues/63) |
| 7 | Possible crash on an all-NaN layer before the bfill/ffill rescue (nhi_chloride.py:44-52) | Contingency — file only if §4.2.5 surfaces it |
| 8 | Housekeeping: stale lint ref, orphaned fixtures, hardcoded local path | [#64](https://github.com/NHFLO/tools/issues/64) |

---

## 9. Implementation progress

Status legend: ☐ not started · ◐ in progress · ☑ done and green.

### Infrastructure

| Item | Status |
|---|---|
| `pyarrow` added to runtime dependencies | ☑ |
| `tests/util.py` — vertex-grid builder etc., verified against real nlmod | ☑ |
| `tests/conftest.py` — hygiene, `vertex_ds`, `gwf_disv` | ☑ |
| pyproject: markers, `MPLBACKEND=Agg`, strict markers, lint-target fix | ☑ |
| GitHub issues for §8 findings | ☑ (#60–#64) |
| Removed empty shells `test_hhnk.py` / `test_nhflo_utils.py` | ☑ |
| `.github/workflows/test.yml` | ☑ |
| README module overview (used-by-09pwnmodel2 vs untested) | ☑ |

### Test files

All green. Suite total: **126 passed, 1 xfailed in ~4 s**; slowest single test 0.59 s.

The suite is independent of the developer's environment: `conftest.pytest_configure` unsets
`NHFLODATA_LOCATION` before collection (so parametrisation and any higher-scoped fixture
also see it unset) and `pytest_unconfigure` restores it. Verified by running the whole suite
three ways — variable unset, set to an existing empty directory, and set to the real mockup
root — all green, and by mutation: with the suppression removed and the variable set, 42
tests fail, led by `test_data_location_env_is_suppressed_for_the_session`, which names the
cause rather than leaving 41 opaque path failures.

| File | Plan § | Tests | Status |
|---|---|---|---|
| `test_panden.py` | 4.3 | 7 | ☑ |
| `test_polder.py` | 4.4 | 4 | ☑ |
| `test_well.py` (extended) | 4.5 | 13 | ☑ |
| `test_major_surface_waters.py` | 4.1 | 7 | ☑ |
| `test_nhi_chloride.py` | 4.2 | 8 | ☑ |
| `test_pwnlayers3_layers.py` | 4.8 | 15 | ☑ (1 xfail, #65) |
| `test_pwnlayers_get_top.py` | 4.7 | 3 | ☑ |
| `test_postprocessing.py` (extended) | 4.6 | 17 | ☑ |
| `test_pwnlayers3_plot.py` | 4.9 | 10 | ☑ |
| `test_nhflodata_contract.py` | 4.10 | 42 | ☑ |
| `test_mf6_smoke.py` | 4.11 | 1 | ☑ |

Every test file was mutation-checked during implementation: a plausible bug was introduced
into the source, the test confirmed to fail, and the source restored. Highlights — the
panden conductance formula reverted to the pre-83365c7 resistance aggregation (caught by
2 tests), the polder fallback conductance zeroed (caught), the northsea fallback guard
removed (caught), `numeric_only` dropped from the well median (caught), the sea-stage
override mask inverted (caught by 3), `extrapolate_ds`'s in-place contract broken (caught),
the layer-split group top misidentified (caught), the `get_top_from_ahn` nearest-donor
metric axis-swapped (caught), the falsy-constant sea guard introduced (caught), the budget
threshold inverted and the budget key renamed (both caught), and the `.hds` head field
mirrored on the x axis (caught by the MF6 run).

### Deviations from the plan

- **§4.3 #8** (pre-gridded frame raises) dropped: unreachable through panden's real code
  path, since `get_oppervlakte_pwn_shapes` always builds its frame from `gpd.read_file`.
  It would have asserted nlmod's own guard, not nhflotools behaviour.
- **§4.4** planned 6 tests, landed 4: the cbot known-answer merged into the partition test
  (parametrized), the NaN-mean and nearest-fill cases merged into one, and the all-NaN
  crash-boundary case dropped as untestable without pinning scipy internals.
- **§4.7** planned 4, landed 3: the no-NaN identity folded into the nearest-donor test.
- **Unit tests use the vertex (DISV) grid** from `tests/util.py` rather than the structured
  grid the plan sketched, so cellids are `(layer, icell2d)` pairs. This matches production
  (09pwnmodel2 is a refined vertex model) and needs no gridgen.
- **§4.2 contingency did not materialise**: the suspected all-NaN-layer crash before the
  bfill/ffill rescue does not reproduce on this environment, so §8 item 7 was not filed.
- **§4.8 REGIS-guard test** revealed a real defect (issue #65) and is committed as
  `xfail(strict=True)`, so it becomes a visible failure the moment the guard is fixed.

### Further findings filed during implementation

| # | Finding |
|---|---|
| [#65](https://github.com/NHFLO/tools/issues/65) | REGIS-completeness guard raises a pandas error in the case it exists for |
| [#66](https://github.com/NHFLO/tools/issues/66) | `chd_ghb_from_major_surface_waters` always returns `ts_sea=None` |
| [#67](https://github.com/NHFLO/tools/issues/67) | Budget check inspects only the incremental budget, never the cumulative |
| [#68](https://github.com/NHFLO/tools/issues/68) | `plot_result_maps` leaks all but the grensvlak figures |
| [#69](https://github.com/NHFLO/tools/issues/69) | Panden SSM guard cannot dedupe on a repeated call against one `gwf` |

Reported but not filed (lower value, recorded here): `polder.py` `si.griddata` raises when
every cell's elevation is NaN; `well.py` silently maps every well to NaN when the flow-tag
namespaces differ, and does not check `locatie` uniqueness; `panden.py` `str.contains`
yields NaN for a NULL `Naam` and raises on the boolean mask; the two copies of
`fix_missings_botms_and_min_layer_thickness` have already diverged in their logging
arithmetic; `_parse_flag_labels` splits on `;` before stripping parentheticals, so a `;`
inside a description silently splits one flag into two labels.
