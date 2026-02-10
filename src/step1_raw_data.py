import requests
import pandas as pd
import io
import json
from pathlib import Path
from datetime import datetime

def fetch_and_save_raw():
    base_dir = Path(__file__).resolve().parent
    raw_dir = base_dir / "data_raw"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    series_ids = {
        'CPALTT01CNM659N': 'cpi_yoy',
        'RBCNBIS': 'reer',
        'CHNSLRTTO02MLM': 'retail_sales',
        'CHNPRINTO01IXPYM': 'ip_yoy',
        'IR3TIB01CNM156N': 'shibor3m'
    }
    
    manifest = []
    
    for s_id, name in series_ids.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s_id}"
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code == 200:
                raw_file = raw_dir / f"fred_{s_id}.csv"
                with open(raw_file, 'w', encoding='utf-8') as f:
                    f.write(resp.text)
                
                df = pd.read_csv(io.StringIO(resp.text))
                df.columns = ['date', 'value']
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                
                manifest.append({
                    'series_id': s_id,
                    'internal_name': name,
                    'download_time': datetime.now().isoformat(),
                    'frequency': 'Monthly',
                    'missing_count': int(df['value'].isna().sum()),
                    'last_date': str(df['date'].iloc[-1])
                })
        except Exception as e:
            print(f"Error {s_id}: {e}")

    with open(raw_dir / "fred_manifest.json", 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print("Raw data and manifest saved.")

if __name__ == "__main__":
    fetch_and_save_raw()
