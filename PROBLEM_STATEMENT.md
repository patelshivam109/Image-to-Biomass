# Problem Statement: Precision Pasture AI

## The Context
In agricultural and livestock management, knowing the exact amount of available forage (pasture biomass) is critical. Farmers, agronomists, and researchers rely on accurate biomass estimations to:
- Determine optimal grazing schedules for livestock (cattle, sheep, etc.).
- Prevent overgrazing, which degrades soil health and ruins future crop yields.
- Calculate supplemental feed requirements to ensure livestock receive adequate nutrition.
- Make data-driven decisions for sustainable farm management and economic forecasting.

## The Problem
Historically, estimating pasture biomass has relied on a highly manual, archaic process known as **"clipping and weighing"**. 
1. A farmer must physically travel out into the pasture.
2. They place a square-meter quadrat on the ground and manually cut all the vegetation inside it.
3. The harvested vegetation is taken back to a facility where it is sorted by species (e.g., grass vs. clover).
4. The vegetation is dried in an oven for 24-48 hours to remove all water weight (as nutritional value is based on "dry matter").
5. The dried matter is finally weighed on a scale to extrapolate the kilograms per hectare (kg/ha) for the entire field.

This traditional method is:
- **Time-Consuming & Labor-Intensive:** It requires hours of physical labor and days of waiting for drying.
- **Costly:** The manual labor and equipment needed for large-scale farm surveys represent a significant financial burden.
- **Unscalable:** Because it is so difficult, farmers can only take a few small samples, which often fail to accurately represent the vast variations across a massive field.
- **Prone to Human Error:** Inconsistent cutting heights and drying times lead to inaccurate data.

## The Solution: Image-to-Biomass Prediction
The **Precision Pasture AI** application aims to completely eliminate the need for manual clipping and weighing by automating the estimation process using Computer Vision (CV) and Deep Learning.

By allowing a user to simply snap a top-down RGB photograph of their pasture using a smartphone or drone and providing basic environmental metadata (like geographic state and canopy height), the AI model can instantly predict the dry biomass yield. 

### Key Objectives Solved:
1. **Instantaneous Results:** What used to take 48 hours now takes less than 3 seconds.
2. **High-Resolution Granularity:** The system separates predictions into specific nutritional categories:
   - Green Dry Matter
   - Dry Clover Biomass (High nutrition)
   - Dry Dead Biomass (Low nutrition)
3. **Extreme Scalability:** Farmers can take dozens of photos across their entire property in minutes, generating a highly accurate, field-wide biomass map without ever picking up a pair of shears.
4. **Visual Interpretability:** Using techniques like Semantic Segmentation and Grad-CAM heatmaps, the application ensures farmers can trust *why* the AI made its prediction, bridging the gap between complex Deep Learning and practical agricultural use.
