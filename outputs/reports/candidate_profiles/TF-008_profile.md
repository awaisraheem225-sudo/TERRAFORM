# TF-008 — Candidate Intelligence Profile

**Area rank:** 8 (by area, not suitability)

## Geographic identification
- Region: Antofagasta
- Centroid latitude: -22.540403
- Centroid longitude: -68.918086
- Area: 43.3025 km²
- Raster pixels: 202

## Terrain characterization
| Metric | Mean | Median |
|---|---:|---:|
| Slope | 2.3468° | 2.3638° |
| Roughness | 15.5173 m | 15.5772 m |
| Local relief | 50.7667 m | 50.8755 m |

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
- Mapped geology area: 43.2302 km²
- Geological map coverage: 99.8331%
- Unmapped area: 0.0723 km²

### Intersecting geological units
| Code | Lithology | Area (km²) | Candidate coverage |
|---|---|---:|---:|
| Qa | grava,arena,limo | 40.3921 | 93.2788% |
| MP1c | conglomerado,arenisca,limolita | 2.8159 | 6.5029% |
| TrJ3 | brecha,basalto,riolita,arenisca | 0.206 | 0.4756% |

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