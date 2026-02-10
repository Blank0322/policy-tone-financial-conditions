import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import norm

def cw_test(actual, pred_base, pred_ext):
    e_base = actual - pred_base
    e_ext = actual - pred_ext
    f_t = e_base**2 - (e_ext**2 - (pred_base - pred_ext)**2)
    X = np.ones(len(f_t))
    model = OLS(f_t, X).fit()
    cw_stat = model.tvalues[0]
    p_value = 1 - norm.cdf(cw_stat)
    return cw_stat, p_value

def dm_test(actual, pred1, pred2):
    e1 = actual - pred1
    e2 = actual - pred2
    d = e1**2 - e2**2
    if np.var(d) == 0: return 0, 1
    dm_stat = np.mean(d) / np.sqrt(np.var(d, ddof=1) / len(d))
    p_value = 2 * (1 - norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value

def run_oos_final_v2():
    base_dir = Path(__file__).resolve().parent
    panel = pd.read_csv(base_dir / "panel_monthly_v2.csv")
    
    # Drop rows with NaNs in targets or best policy variable
    targets = ['cpi_yoy', 'dcpi_yoy', 'gap_cpi']
    best_pv = 'tone_logit'
    panel_clean = panel.dropna(subset=targets + [best_pv]).reset_index(drop=True)
    
    window = 120
    p = 12
    all_oos_results = []
    all_cw_results = []
    
    sub_periods = {
        'Full Sample': (window, len(panel_clean) - 1),
        'Pre-COVID': (window, panel_clean[panel_clean['month'] < '2020-01'].index[-1]),
        'Post-2020': (panel_clean[panel_clean['month'] >= '2020-01'].index[0], len(panel_clean) - 1)
    }

    for target in targets:
        print(f"OOS for {target}...")
        y = panel_clean[target].values
        x = panel_clean[best_pv].values
        
        preds_ar = np.full(len(panel_clean), np.nan)
        preds_arx = np.full(len(panel_clean), np.nan)
        
        for t in range(window, len(panel_clean) - 1):
            train_y = y[t-window:t]
            train_x = x[t-window:t]
            
            def create_lags(data, lags):
                n = len(data)
                X = np.ones((n - lags, lags + 1))
                for i in range(lags):
                    X[:, i+1] = data[lags-i-1 : n-i-1]
                return X, data[lags:]

            X_base, y_train_endog = create_lags(train_y, p)
            model_base = OLS(y_train_endog, X_base).fit()
            pred_base = model_base.predict([1] + list(train_y[-p:][::-1]))[0]
            
            X_ext = np.column_stack([X_base, train_x[p-1:-1]])
            model_ext = OLS(y_train_endog, X_ext).fit()
            pred_ext = model_ext.predict([1] + list(train_y[-p:][::-1]) + [train_x[-1]])[0]
            
            preds_ar[t] = pred_base
            preds_arx[t] = pred_ext
            
        for period_name, (start, end) in sub_periods.items():
            if start >= end: continue
            p_actual = y[start+1:end+1] # t+1
            p_ar = preds_ar[start:end]
            p_arx = preds_arx[start:end]
            
            mask = ~np.isnan(p_ar)
            p_actual, p_ar, p_arx = p_actual[mask], p_ar[mask], p_arx[mask]
            
            if len(p_actual) < 5: continue
            
            rmse_ar = np.sqrt(mean_squared_error(p_actual, p_ar))
            rmse_arx = np.sqrt(mean_squared_error(p_actual, p_arx))
            
            cw_stat, cw_p = cw_test(p_actual, p_ar, p_arx)
            dm_stat, dm_p = dm_test(p_actual, p_ar, p_arx)
            
            all_oos_results.append({'Target': target, 'Period': period_name, 'RMSE_AR': rmse_ar, 'RMSE_ARX': rmse_arx, 'Improvement': (rmse_ar-rmse_arx)/rmse_ar*100})
            all_cw_results.append({'Target': target, 'Period': period_name, 'CW_P': cw_p, 'DM_P': dm_p})

    pd.DataFrame(all_oos_results).to_csv(base_dir / "tables/oos_forecast_cpi_v2.csv", index=False)
    pd.DataFrame(all_cw_results).to_csv(base_dir / "tables/clark_west_cpi_v2.csv", index=False)
    print("Final OOS completed.")

if __name__ == "__main__":
    run_oos_final_v2()
