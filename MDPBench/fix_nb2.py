import json
import re

with open('/home/zhibolin/MultimodalOCR/MDPBench/tools/generate_result_tables.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if "pages = (" in line:
                pass
            if ".join(" in line and "n_formula" in line:
                cell['source'][i] = line.replace('n_formula,', 'pd.Series(n_formula, name="n_formula"),')
            if ".join(" in line and "n_text" in line:
                cell['source'][i] = line.replace('n_text,', 'pd.Series(n_text, name="n_text"),')
            if ".join(" in line and "n_table" in line:
                cell['source'][i] = line.replace('n_table,', 'pd.Series(n_table, name="n_table"),')
            if ".join(" in line and "text_s," in line:
                cell['source'][i] = line.replace('text_s,', 'pd.Series(text_s, name="s_text"),')
            if ".join(" in line and "formula_s," in line:
                cell['source'][i] = line.replace('formula_s,', 'pd.Series(formula_s, name="s_formula"),')
            if ".join(" in line and "table_s," in line:
                cell['source'][i] = line.replace('table_s,', 'pd.Series(table_s, name="s_table"),')

with open('/home/zhibolin/MultimodalOCR/MDPBench/tools/generate_result_tables.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print("done")
