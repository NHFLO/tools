# NHFLO tools
Deze repository bevat de `nhflotools` Python package met tools om NHFLO modellen mee te maken. De package wordt gebruikt in modelscripts uit de [NHFLO models](https://github.com/NHFLO/tools) repository.

Note that the content of this repository is available under a not-so-permissive open-source licence: GNU AGPLv3. Please have a look at [choose a license](https://choosealicense.com/licenses/agpl-3.0/) for the key conditions and limitations of this license before getting started.

## Installatie
De `nhflotools` package kan worden geinstallerd door deze repository te clonen en vervolgens `pip install -e .  --config-settings editable_mode=strict` te runnen vanuit de package map.

## Overzicht van de modules

Het modelscript [`09pwnmodel2`](https://github.com/NHFLO/models/tree/main/modelscripts/09pwnmodel2) is
het actieve PWN-model en gebruikt maar een deel van deze package. Die modules vormen de
onderhouden kern: ze worden in CI met pytest getest. De overige modules zijn niet in gebruik
door 09pwnmodel2 en zijn ongetest — behandel ze als legacy of werk-in-uitvoering.

### Gebruikt door 09pwnmodel2 (getest in CI)

| Module | Wat het doet | Gebruikt in 09pwnmodel2 |
|---|---|---|
| `major_surface_waters.py` | Grote RWS-oppervlaktewateren: Noordzee als CHD (met zeespiegel-tijdreeks en 18.000 mg/l chloride), IJsselmeer/Markermeer/Noordzeekanaal als GHB | `get_chd_ghb_data_from_major_surface_waters`, `chd_ghb_from_major_surface_waters` |
| `nhi_chloride.py` | NHI-chlorideconcentratie interpoleren naar het modelgrid als begintoestand voor transport | `get_nhi_chloride_concentration` |
| `panden.py` | Infiltratiepanden (ICAS/IKIEF) als RIV-package | `riv_from_oppervlakte_pwn` |
| `polder.py` | Polderpeilgebieden van HHNK als DRN-package, met maaiveld als terugval | `drn_from_waterboard_data` |
| `postprocessing.py` | Waterbalanscontrole, modeluitvoer inlezen, grensvlakken zoet/brak en resultaatkaarten | `check_budget_discrepancy`, `add_output_to_ds`, `plot_result_maps` |
| `pwnlayers/layers.py` | Alleen de maaiveldhoogte uit AHN, inclusief opvullen bij oppervlaktewater en zee | `get_top_from_ahn` |
| `pwnlayers3/layers.py` | Het PWN-lagenmodel (v3): botm, kh en kv uit de bodemlagenkartering, samengevoegd met REGIS | `get_pwn_layer_model` |
| `pwnlayers3/plot.py` | Diagnostische dwarsdoorsneden van het lagenmodel | `plot_diagnostic_cross_sections` |
| `well.py` | Winnings- en infiltratieputten van PWN en Tata Steel | `get_wells_pwn_dataframe`, `get_wells_tata_dataframes` |
| `pwnlayers/merge_layer_models.py` | Twee lagenmodellen samenvoegen via een koppeltabel, met overgangszone | indirect, via `pwnlayers3` |
| `pwnlayers/utils.py` | Ontbrekende en elkaar kruisende laagbodems repareren | indirect, via `merge_layer_models` |

### Niet gebruikt door 09pwnmodel2 (ongetest)

| Module | Wat het doet |
|---|---|
| `bergen_utils.py` | Lagenmodel en oppervlaktewater voor het oudere Bergen-model |
| `berging_utils.py` | Bergingscoëfficiënten per perceel, afgeleid van nabijgelegen oppervlaktewater |
| `bofek.py` | BOFEK-bodemprofielen en de bijbehorende berging |
| `cropfactor.py` | Gewasfactoren toepassen op verdamping |
| `geoconverter/` | Command-line tool om geodata naar het NHFLO-dataformaat te converteren |
| `hhnk.py` | Peilbuisreeksen ophalen uit de FEWS-webservice van HHNK |
| `nhflo_utils.py` | Verzameling oudere plot- en gridhulpfuncties |
| `pwnlayers/io.py` | Inlezen van de Mensink- en Bergen-bodemlagen (lagenmodel v1) |
| `utils.py` | Lokale kopie van nlmod's MODFLOW-binaries-afhandeling |

## Tests

De suite telt 127 tests en draait in ongeveer 4 seconden. De tests gebruiken uitsluitend
kleine synthetische modellen — geen netwerk, geen gridgen en geen grote datasets — zodat
ze op elke pull request in CI meedraaien. Alle live webservices (HHNK, REGIS, RWS,
achtergrondkaarten) worden per naam gemonkeypatcht; de rest van nlmod draait echt, waardoor
de suite meteen dienstdoet als compatibiliteitscanary voor `nlmod@dev`.

`NHFLODATA_LOCATION` wordt in `pyproject.toml` via pytest-env op leeg gezet, zodat de tests
altijd tegen de meegeleverde mockup-data draaien — ook op een machine waar een echte
datamap is gekoppeld. Dat geldt alleen binnen pytest; een modelscript gebruikt gewoon jouw
eigen datamap.

De enige uitzondering is één test met de marker `mf6`: die draait MODFLOW echt op een
model van 3x3x2 cellen en controleert de waterbalans plus een analytische oplossing. De
binaries worden zo nodig automatisch door nlmod gedownload en daarna hergebruikt.

```bash
hatch run test:test           # de hele suite
pytest -m "not mf6"           # zonder de MODFLOW-run
```

De opzet en onderbouwing van de suite staan in [`TEST_PLAN.md`](TEST_PLAN.md), inclusief
de mutatietests waarmee per bestand is aangetoond dat de tests echte fouten vangen.