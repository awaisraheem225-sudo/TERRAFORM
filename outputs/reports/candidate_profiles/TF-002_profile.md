# TF-002 — Candidate Intelligence Profile

**Area rank:** 2 (by area, not suitability)

## Geographic identification
- Region: Antofagasta
- Centroid latitude: -22.9095
- Centroid longitude: -68.809654
- Area: 80.3884 km²
- Raster pixels: 375

## Terrain characterization
| Metric | Mean | Median |
|---|---:|---:|
| Slope | 2.2918° | 2.2659° |
| Roughness | 15.2179 m | 15.1213 m |
| Local relief | 50.1085 m | 49.9465 m |

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
- Mapped geology area: 80.078 km²
- Geological map coverage: 99.6139%
- Unmapped area: 0.3104 km²

### Intersecting geological units
| Code | Lithology | Area (km²) | Candidate coverage |
|---|---|---:|---:|
| MP1c | conglomerado,arenisca,limolita | 51.9783 | 64.659% |
| Qa | grava,arena,limo | 24.2869 | 30.212% |
| CPg | granito,granodiorita,tonalita,diorita | 3.7081 | 4.6128% |
| CP3 | andesita,riolita,arenisca,caliza | 0.5541 | 0.6893% |

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