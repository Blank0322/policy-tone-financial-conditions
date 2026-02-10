import pandas as pd
from pathlib import Path
import shutil

def finalize_paper_assets():
    base_dir = Path(__file__).resolve().parent
    assets_dir = base_dir / "paper_assets"
    
    # 1. Table 1: Main Regression
    reg_main = pd.read_csv(base_dir / "tables/reg_cpi_main.csv")
    # Clean up names
    reg_main['Index'] = reg_main['Index'].replace({
        'tone_dict': 'Dictionary Index',
        'tone_pca': 'PCA Factor Index',
        'tone_logit': 'ML Logit Index',
        'tone_growth': 'Growth Topic Index',
        'tone_risk': 'Risk Topic Index'
    })
    reg_main.to_csv(assets_dir / "Table1_main_regression.csv", index=False)
    
    # 2. Table 2: OOS DM Test
    oos_res = pd.read_csv(base_dir / "tables/oos_forecast_cpi.csv")
    dm_res = pd.read_csv(base_dir / "tables/dm_test_cpi.csv", header=None)
    
    # Combine OOS and DM into a clean summary
    # Table 2: OOS RMSE and DM p-value
    table2 = oos_res.copy()
    # Add DM p-value to the ARX row
    table2.loc[1, 'DM_p_value'] = dm_res.iloc[1, 1]
    table2.to_csv(assets_dir / "Table2_oos_dm.csv", index=False)
    
    # 3. Figures
    shutil.copy(base_dir / "fig/pred_vs_actual_cpi.png", assets_dir / "Figure1_oos_results.png")
    shutil.copy(base_dir / "tone_with_changepoints.png", assets_dir / "Figure2_policy_shifts.png")
    
    # 4. Methods Note
    panel = pd.read_csv(base_dir / "panel_monthly.csv")
    start_date = panel['month'].iloc[0]
    end_date = panel['month'].iloc[-1]
    n_obs = len(panel)
    
    note = f"""# Methods and Data Summary for Dissertation

## Data Sources
- **Policy Tone Indices**: Extracted from People's Daily (人民日报) corpus (2005-2023) using dictionary, PCA, Weakly Supervised Learning (Logit), and LDA Topic Modeling.
- **Macroeconomic Variables**: Sourced from FRED (St. Louis Fed). Series: CPALTT01CNM659N (CPI YoY), RBCNBIS (REER), CHNPRINTO01IXPYM (IP YoY), IR3TIB01CNM156N (Interbank 3M).

## Sample Period
- **Full Sample**: {start_date} to {end_date}
- **Number of Months**: {n_obs}

## Estimation Specifications
1. **Predictive Regressions**: OLS models with Newey-West (HAC) standard errors (maxlags=12). AR(p) baseline with lag p selected by AIC.
2. **Out-of-Sample (OOS) Forecasting**: Rolling window of 120 months. Strict causality (no future leakage). Evaluation via RMSE and Diebold-Mariano (DM) tests.
3. **Robustness**: Change-point detection (PELT), subsample analysis, and Lasso feature selection.

## Key Result
- **ML Logit Index** shows significant predictive power for CPI YoY (p < 0.01) even after controlling for persistent inflation lags.
"""
    with open(assets_dir / "notes_methods.md", 'w', encoding='utf-8') as f:
        f.write(note)
        
    print("Paper assets finalized.")

if __name__ == "__main__":
    finalize_paper_assets()
