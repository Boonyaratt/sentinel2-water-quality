# Figure Gallery

Figure numbers refer to the research manuscript. The file names `figure1` and `figure2` in some notebooks refer to Figs. 4 and 5 here.

## Fig. 1 · Research workflow

![Research workflow from data acquisition to explainable predictions](../figures/01-research-workflow.png)

## Fig. 2 · Sampling locations

![Study area and water-quality sampling locations](../figures/02-sampling-locations.png)

## Fig. 3 · Representative prediction maps

![Representative water-quality prediction maps](../figures/03-representative-maps.png)

The maps show Thale Luang (WB02, 19 July 2024), Lower Lang Suan River (WB07, 20 January 2025), and Pak Phanang River (WB05, 4 July 2025).

## Fig. 4 · Effect of contextual predictors

Within each site, the prediction difference is **S2 + Context minus S2 only**, displayed on common water pixels under the analysis mask.

![Thale Luang comparison](../figures/04-wb02.png)

![Lower Lang Suan River comparison](../figures/04-wb07.png)

![Pak Phanang River comparison](../figures/04-wb05.png)

## Fig. 5 · Change between two acquisitions

![Thale Luang predictions on 4 and 19 June 2024 and their differences](../figures/05-two-date-change.png)

The contrast compares 19 June with 4 June 2024 on the common valid-water mask. The scene-selection record identifies contextual contrast and usable coverage as the selection criteria. The changes in the predicted targets are outcomes of that comparison.

## Fig. 6 · Global SHAP attribution

![Global SHAP for the three locked target models](../figures/06-global-shap.png)

## Fig. 7 · Regional SHAP attribution

![Feature-group attribution by waterbody and target](../figures/07-regional-shap.png)

This analysis uses the archived v1 feature and prediction grid. Its regional attribution is not computed from the polygon rasters used for Figs. 3–4.
