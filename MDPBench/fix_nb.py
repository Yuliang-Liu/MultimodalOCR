import json

with open('/home/zhibolin/MultimodalOCR/MDPBench/tools/generate_result_tables.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if 'df[f"n_{k}"] = df.get(f"n_{k}", 0).fillna(0).astype(float)' in line:
                cell['source'][i] = '        df[f"n_{k}"] = df[f"n_{k}"].fillna(0).astype(float) if f"n_{k}" in df else 0.0\n'
            if 'df[f"s_{k}"] = df.get(f"s_{k}", np.nan).astype(float)' in line:
                cell['source'][i] = '        df[f"s_{k}"] = df[f"s_{k}"].astype(float) if f"s_{k}" in df else np.nan\n'

with open('/home/zhibolin/MultimodalOCR/MDPBench/tools/generate_result_tables.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print("done")
