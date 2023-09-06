
# NHFLO tools
Deze repository bevat de `nhflotools` Python package met tools om NHFLO modellen mee te maken. De package wordt gebruikt in modelscripts uit de [NHFLO models](https://github.com/ArtesiaWater/NHFLO_models) repository.

## Installatie
De `nhflotools` package kan worden geinstallerd door deze repository te clonen en vervolgens `pip install -e .` te runnen vanuit de package map.

## Ontwikkelaars
### Installatie
 - `conda create --name nhflo python=3.10`
 - `conda activate nhflo`
 - `pip install -e ".[dev]"`

### precommit
 - `hatch run format`
 - `hatch run lint`
 - `hatch run test`