# Multi-Gap Analyses

Python scripts for normative gap analyses of UK Biobank brain features, clinical and diagnostic associations, model benchmarking, and sensitivity analyses using FreeSurfer and functional connectivity data.

## Scripts

| Script | Purpose                                                                                                  |
|---|----------------------------------------------------------------------------------------------------------|
| [`1_compute_normative_gaps.py`](1_compute_normative_gaps.py) | Fit normative prediction models and calculate standardized brain-feature gaps.                           |
| [`2_compute_gap_associations.py`](2_compute_gap_associations.py) | Test gap associations with diagnoses and clinical scores; generate association and attenuation figures.  |
| [`3_benchmark_normative_models.py`](3_benchmark_normative_models.py) | Evaluate out-of-fold prediction performance and coefficient-based feature importance.                    |
| [`4_1_largest_site_sensitivity.py`](4_1_largest_site_sensitivity.py) | Test robustness using the largest imaging site as a holdout.                                             |
| [`4_2_modality_sensitivity.py`](4_2_modality_sensitivity.py) | Compare associations before and after adjustment for leave-one-out modality summaries.                   |
| [`4_3_questionnaire_missingness_sensitivity.py`](4_3_questionnaire_missingness_sensitivity.py) | Refit under sctrict questionnaire-based definitions and compare with main analysis results.                       |
| [`5_gap_gap_associations_multivariate.py`](5_gap_gap_associations_multivariate.py) | Calculate pairwise gap correlations and multivariate associations with clinical and diagnostic measures. |

## Dependencies

```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib
```
