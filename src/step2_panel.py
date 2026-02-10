import pandas as pd
from pathlib import Path

def create_panel():
    base_dir = Path(__file__).resolve().parent
    raw_dir = base_dir / "data_raw"
    indices = pd.read_csv(base_dir / "final_policy_indices_v3.csv")
    
    series_ids = {
        'CPALTT01CNM659N': 'cpi_yoy',
        'RBCNBIS': 'reer',
        'CHNSLRTTO02MLM': 'retail_sales',
        'CHNPRINTO01IXPYM': 'ip_yoy',
        'IR3TIB01CNM156N': 'shibor3m'
    }
    
    macro_list = []
    for s_id, name in series_ids.items():
        df = pd.read_csv(raw_dir / f"fred_{s_id}.csv")
        df.columns = ['month', name]
        df['month'] = pd.to_datetime(df['month']).dt.strftime('%Y-%m')
        df[name] = pd.to_numeric(df[name], errors='coerce')
        macro_list.append(df.set_index('month'))
        
    panel = pd.concat(macro_list, axis=1)
    panel = panel.join(indices.set_index('month'), how='inner')
    panel = panel.sort_index().reset_index()
    
    panel.to_csv(base_dir / "panel_monthly.csv", index=False)
    
    # Summary
    summary = panel.describe().transpose()
    summary['missing_rate'] = panel.isnull().mean()
    
    with open(base_dir / "panel_summary.md", 'w', encoding='utf-8') as f:
        f.write("# Panel Monthly Summary\n\n")
        f.write(f"Sample Period: {panel['month'].iloc[0]} to {panel['month'].iloc[-1]}\n")
        f.write(f"Total Observations: {len(panel)}\n\n")
        f.write(summary.to_markdown())
        
    print("Panel created and summarized.")

if __name__ == "__main__":
    create_panel()
