# TF-003 — Candidate Intelligence Profile

**Area rank:** 3 (by area, not suitability)

## Geographic identification
- Region: Antofagasta
- Centroid latitude: -22.744204
- Centroid longitude: -68.903604
- Area: 74.1717 km²
- Raster pixels: 346

## Terrain characterization
| Metric | Mean | Median |
|---|---:|---:|
| Slope | 2.3305° | 2.3299° |
| Roughness | 15.4453 m | 15.3439 m |
| Local relief | 49.2745 m | 48.7063 m |

## Jezero reference interval membership
- slope_median_within_strict_interval: **True**
- roughness_median_within_strict_interval: **True**
- local_relief_median_within_strict_interval: **True**

These are median-value interval checks, not a validated similarity score.

## Screening statistics
- Broad overlap: 100.0%
- Moderate overlap: 100.0%
- These are inherited from the nested screening workflow, not independent validation.

## Geological characterization
- Mapped geology area: 73.9692 km²
- Geological map coverage: 99.727%
- Unmapped area: 0.2025 km²

### Intersecting geological units
| Code | Lithology | Area (km²) | Candidate coverage |
|---|---|---:|---:|
| MP1c | conglomerado,arenisca,limolita | 60.0856 | 81.0088% |
| CPg | granito,granodiorita,tonalita,diorita | 10.1486 | 13.6826% |
| Qa | grava,arena,limo | 3.492 | 4.708% |
| M1c | grava,arena,limo,ignimbrita | 0.3373 | 0.4547% |
| CP3 | andesita,riolita,arenisca,caliza | 0.3124 | 0.4211% |
| PE3a | dacita,riolita,volcanico piroclastico | 0.0634 | 0.0855% |

## Scientific limitations
- Candidate ranking is by connected-region area, not scientific suitability.
- Terrain screening uses empirical Jezero percentile intervals, not validated analogue thresholds.
- Terrain compatibility does not establish geological or mineralogical equivalence.
- The Earth elevation source is Copernicus DEM GLO-30, a digital surface model.
- Mars terrain reference is derived from MOLA gridded elevation data.
- The common working resolution is approximately 463 metres.
- Earth and Mars use separate local projected coordinate systems.
- The SERNAGEOMIN geological map is regional-scale mapping at approximately 1:1,000,000.
- Geological map coverage does not establish field-verified lithology or mineral composition.
- Earth cannot reproduce Martian gravity, atmospheric pressure, radiation environment or atmospheric chemistry.
- The broad and moderate overlap percentages describe the screening workflow, not independent validation.

## Data interpretation
This profile characterizes terrain and regional geological mapping. It does not establish mineralogical equivalence with Mars or prove that the candidate is an operational analogue.

---
*TERRAFORM — Planetary Analog Intelligence*