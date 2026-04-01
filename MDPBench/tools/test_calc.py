import numpy as np
import pandas as pd
from pathlib import Path
from calculate_scores import load_text_scores_and_counts, load_formula_scores_and_counts, load_table_scores_and_counts, lang_from_img_id, compute_present_tasks_overall
result_folder = Path("../result")
prefix = "Gemini3-pro-preview_demo_result"
match_name = "quick_match"
split = "All"
text_s, n_text = load_text_scores_and_counts(result_folder=result_folder, ocr_prefix=prefix, match_name=match_name, split_tag=split)
formula_s, n_formula = load_formula_scores_and_counts(result_folder=result_folder, ocr_prefix=prefix, match_name=match_name, split_tag=split)
table_s, n_table = load_table_scores_and_counts(result_folder=result_folder, ocr_prefix=prefix, match_name=match_name, split_tag=split, metric="TEDS")

print("n_text empty", n_text.empty, "n_formula empty", n_formula.empty, "n_table empty", n_table.empty)
pages = pd.DataFrame()
if not text_s.empty:
    pages = text_s.join(n_text, how="outer")
if not formula_s.empty:
    pages = pages.join(formula_s, how="outer").join(n_formula, how="outer") if not pages.empty else formula_s.join(n_formula, how="outer")
if not table_s.empty:
    pages = pages.join(table_s, how="outer").join(n_table, how="outer") if not pages.empty else table_s.join(n_table, how="outer")

print("pages empty", pages.empty)
pages.index.name = "img_id"
pages["lang"] = [lang_from_img_id(i) for i in pages.index]
pages_scheme2 = compute_present_tasks_overall(pages, empty_policy="nan")
overall_score = pages_scheme2["overall_i_scheme2"].mean(skipna=True)
print("overall mean", overall_score)
print(pages_scheme2["overall_i_scheme2"].head())
