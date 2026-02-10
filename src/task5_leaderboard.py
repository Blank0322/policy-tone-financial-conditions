import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

def run_leaderboard():
    base_dir = Path(__file__).resolve().parent
    panel = pd.read_csv(base_dir / "panel_monthly_v2.csv")
    
    # Target candidates
    # Create diffs for others
    panel['dreer'] = panel['reer'].diff(1)
    panel['dip_yoy'] = panel['ip_yoy'].diff(1)
    panel['dshibor3m'] = panel['shibor3m'].diff(1)
    
    targets = ['dcpi_yoy', 'dreer', 'dip_yoy', 'dshibor3m']
    policy_var = 'tone_logit'
    p = 12
    
    leaderboard = []
    
    for target in targets:
        df = panel[[target, policy_var]].copy()
        for i in range(1, p+1):
            df[f'lag{i}'] = df[target].shift(i)
        df['p_lag1'] = df[policy_var].shift(1)
        df = df.dropna()
        
        # 1. In-sample Wald
        X = sm.add_constant(df[[f'lag{i}' for i in range(1, p+1)] + ['p_lag1']])
        res = OLS(df[target], X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
        p_val = res.pvalues['p_lag1']
        
        # 2. Score Calculation
        score = (1 - p_val) * 100 # Simple score
        
        leaderboard.append({
            'Target': target,
            'P_Val': p_val,
            'Beta': res.params['p_lag1'],
            'Score': score
        })
        
    res_df = pd.DataFrame(leaderboard).sort_values('Score', ascending=False)
    res_df.to_csv(base_dir / "tables/macro_target_leaderboard.csv", index=False)
    print("Leaderboard complete.")

if __name__ == "__main__":
    run_leaderboard()
