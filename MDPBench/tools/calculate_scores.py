import argparse
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

def _page_id_from_indexed_key(key: str) -> str:
    m = re.match(r"^(.*)_\[(\d+)\]$", key)
    return m.group(1) if m else key

def _first_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("None exists: " + " | ".join(str(p) for p in paths))

def iter_json_array(path: Path, *, chunk_size: int = 1 << 20):
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        buf = ""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                raise ValueError(f"Empty/invalid json array: {path}")
            buf += chunk
            i = buf.find("[")
            if i != -1:
                buf = buf[i + 1 :]
                break
        while True:
            buf = buf.lstrip()
            if not buf:
                chunk = f.read(chunk_size)
                if not chunk:
                    raise ValueError(f"Unexpected EOF in json array: {path}")
                buf += chunk
                continue
            if buf.startswith(","):
                buf = buf[1:]
                continue
            if buf.startswith("]"):
                return
            while True:
                try:
                    obj, end = decoder.raw_decode(buf)
                    break
                except json.JSONDecodeError:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        raise
                    buf += chunk
            yield obj
            buf = buf[end:]

def load_text_scores_and_counts(*, result_folder: Path, ocr_prefix: str, match_name: str, split_tag: str = "All"):
    try:
        per_page_edit = _first_existing([
            result_folder / f"{ocr_prefix}_{match_name}_text_block_{split_tag}_per_page_edit.json",
            result_folder / f"{ocr_prefix}_{match_name}_text_block_per_page_edit.json",
        ])
    except FileNotFoundError:
        return pd.DataFrame(), pd.Series(dtype=int)

    text_block_result_path = _first_existing([
        result_folder / f"{ocr_prefix}_{match_name}_text_block_result.json",
    ])

    with per_page_edit.open("r", encoding="utf-8") as f:
        page_edit = json.load(f)
    
    want_is_digit = None
    if split_tag == "digit":
        want_is_digit = True
    elif split_tag == "photo":
        want_is_digit = False
    
    valid_img_ids = set()
    counts = defaultdict(int)
    for item in iter_json_array(text_block_result_path):
        if want_is_digit is not None:
            is_orig = item.get("is_digit", True) 
            if bool(is_orig) != want_is_digit:
                continue
        img_id = item.get("img_id")
        if isinstance(img_id, str) and img_id:
            counts[img_id] += 1
            valid_img_ids.add(img_id)

    filtered_page_edit = {k: v for k, v in page_edit.items() if k in valid_img_ids}
    if not filtered_page_edit:
        return pd.DataFrame(), pd.Series(dtype=int)

    text_df = pd.DataFrame({"img_id": list(filtered_page_edit.keys()), "text_edit": list(filtered_page_edit.values())})
    text_df["s_text"] = (1.0 - text_df["text_edit"].astype(float)).clip(0, 1)
    text_df = text_df.set_index("img_id")["s_text"].to_frame()

    n_text = pd.Series(counts, name="n_text").astype(int)
    n_text = n_text[n_text.index.isin(text_df.index)]
    return text_df, n_text

def load_formula_scores_and_counts(*, result_folder: Path, ocr_prefix: str, match_name: str, split_tag: str = "All"):
    try:
        per_sample_cdm = _first_existing([
            result_folder / f"{ocr_prefix}_{match_name}_display_formula_{split_tag}_per_sample_CDM.json",
            result_folder / f"{ocr_prefix}_{match_name}_display_formula_per_sample_CDM.json",
        ])
    except FileNotFoundError:
        return pd.DataFrame(), pd.Series(dtype=int)

    formula_result_path = _first_existing([
        result_folder / f"{ocr_prefix}_{match_name}_display_formula_result.json",
    ])

    with per_sample_cdm.open("r", encoding="utf-8") as f:
        d = json.load(f)

    want_is_digit = None
    if split_tag == "digit":
        want_is_digit = True
    elif split_tag == "photo":
        want_is_digit = False
    
    img_is_orig_map = {}
    for item in iter_json_array(formula_result_path):
        img_id = item.get("img_id")
        if img_id and "is_digit" in item:
            img_is_orig_map[img_id] = item["is_digit"]
        elif img_id and img_id not in img_is_orig_map:
            img_is_orig_map[img_id] = True

    sums = defaultdict(float)
    cnts = defaultdict(int)
    for key, val in d.items():
        if val is None:
            continue
        page_id = _page_id_from_indexed_key(str(key))
        if want_is_digit is not None:
            is_orig = img_is_orig_map.get(page_id, True)
            if bool(is_orig) != want_is_digit:
                continue

        sums[page_id] += float(val)
        cnts[page_id] += 1

    if not sums:
        return pd.DataFrame(), pd.Series(dtype=int)

    s_formula = pd.Series({k: sums[k] / cnts[k] for k in cnts}, name="s_formula").astype(float)
    n_formula = pd.Series(cnts, name="n_formula").astype(int)
    return s_formula.to_frame(), n_formula

def load_table_scores_and_counts(*, result_folder: Path, ocr_prefix: str, match_name: str, split_tag: str = "All", metric: str = "TEDS"):
    try:
        per_table = _first_existing([
            result_folder / f"{ocr_prefix}_{match_name}_table_{split_tag}_per_table_TEDS.json",
            result_folder / f"{ocr_prefix}_{match_name}_table_per_table_TEDS.json",
        ])
    except FileNotFoundError:
        return pd.DataFrame(), pd.Series(dtype=int)

    table_result_path = _first_existing([
        result_folder / f"{ocr_prefix}_{match_name}_table_result.json",
    ])
    
    want_is_digit = None
    if split_tag == "digit":
        want_is_digit = True
    elif split_tag == "photo":
        want_is_digit = False

    img_is_orig_map = {}
    for item in iter_json_array(table_result_path):
        img_id = item.get("img_id")
        if img_id:
            img_is_orig_map[img_id] = item.get("is_digit", True)

    with per_table.open("r", encoding="utf-8") as f:
        d = json.load(f)

    sums = defaultdict(float)
    cnts = defaultdict(int)
    for key, metrics in d.items():
        if not isinstance(metrics, dict):
            continue
        score = metrics.get(metric)
        if score is None:
            continue
        
        page_id = _page_id_from_indexed_key(str(key))
        if want_is_digit is not None:
            is_orig = img_is_orig_map.get(page_id, True)
            if bool(is_orig) != want_is_digit:
                continue

        sums[page_id] += float(score)
        cnts[page_id] += 1

    if not sums:
        return pd.DataFrame(), pd.Series(dtype=int)

    s_table = pd.Series({k: sums[k] / cnts[k] for k in cnts}, name="s_table").astype(float)
    n_table = pd.Series(cnts, name="n_table").astype(int)
    return s_table.to_frame(), n_table

def lang_from_img_id(img_id: str) -> str:
    return img_id.split("_", 1)[0] if isinstance(img_id, str) and "_" in img_id else "unknown"

def compute_present_tasks_overall(pages: pd.DataFrame, *, empty_policy: str = "nan") -> pd.DataFrame:
    df = pages.copy()
    for k in ["text", "formula", "table"]:
        df[f"n_{k}"] = df.get(f"n_{k}", 0).fillna(0).astype(float)
        df[f"s_{k}"] = df.get(f"s_{k}", np.nan).astype(float)

    use_text = (df["n_text"] > 0) & df["s_text"].notna()
    use_formula = (df["n_formula"] > 0) & df["s_formula"].notna()
    use_table = (df["n_table"] > 0) & df["s_table"].notna()

    denom = use_text.astype(int) + use_formula.astype(int) + use_table.astype(int)
    numer = (
        df["s_text"].fillna(0) * use_text.astype(int)
        + df["s_formula"].fillna(0) * use_formula.astype(int)
        + df["s_table"].fillna(0) * use_table.astype(int)
    )

    overall_i = numer / denom.replace({0: np.nan})
    if empty_policy == "zero":
        overall_i = overall_i.fillna(0.0)

    df["overall_i_scheme2"] = overall_i.clip(0, 1)
    df["present_task_cnt"] = denom
    return df

def main():
    parser = argparse.ArgumentParser(description="Compute OCR metrics matching generate_result_tables.ipynb cell 3.")
    parser.add_argument("prefix", type=str, help="Model result prefix, e.g., Gemini-3-pro-preview_private")
    parser.add_argument("--result_folder", type=str, default="../result", help="Path to the result folder")
    parser.add_argument("--match_name", type=str, default="quick_match", help="Match name used in result files")
    
    args = parser.parse_args()

    result_folder = Path(args.result_folder)
    prefix = args.prefix
    match_name = args.match_name

    split_tags = ["All", "digit", "photo"]
    split_rename_map = {"digit": "Digit.", "photo": "Photo.", "All": "All"}
    latin_langs = ["DE", "EN", "ES", "FR", "ID", "IT", "NL", "PT", "VI"]
    non_latin_langs = ["AR", "HI", "JP", "KO", "RU", "TH", "ZH", "ZH-T"]

    model_data = {}
    
    print(f"==================================================")
    print(f"Model Summary For: {prefix}")
    print(f"==================================================\n")

    for split in split_tags:
        try:
            text_s, n_text = load_text_scores_and_counts(
                result_folder=result_folder, ocr_prefix=prefix, match_name=match_name, split_tag=split
            )
            formula_s, n_formula = load_formula_scores_and_counts(
                result_folder=result_folder, ocr_prefix=prefix, match_name=match_name, split_tag=split
            )
            table_s, n_table = load_table_scores_and_counts(
                result_folder=result_folder, ocr_prefix=prefix, match_name=match_name, split_tag=split, metric="TEDS"
            )

            pages = pd.DataFrame()
            if not text_s.empty:
                pages = text_s.join(n_text, how="outer")
            if not formula_s.empty:
                pages = pages.join(formula_s, how="outer").join(n_formula, how="outer") if not pages.empty else formula_s.join(n_formula, how="outer")
            if not table_s.empty:
                pages = pages.join(table_s, how="outer").join(n_table, how="outer") if not pages.empty else table_s.join(n_table, how="outer")

            if not pages.empty:
                pages.index.name = "img_id"
                pages["lang"] = [lang_from_img_id(i) for i in pages.index]

                pages_scheme2 = compute_present_tasks_overall(pages, empty_policy="nan")
                
                # Group by Language for "All" split
                if split == "All":
                    lang_group = pages_scheme2.groupby("lang")["overall_i_scheme2"].mean()
                    for lang, score in lang_group.items():
                        lang_up = lang.upper()
                        if lang_up == "ZH-CHT":
                            lang_up = "ZH-T"
                        model_data[(lang_up, "")] = round(score * 100, 1)

                # Compute Overall for this split
                overall_score = pages_scheme2["overall_i_scheme2"].mean(skipna=True)
                model_data[("Overall", split_rename_map[split])] = round(overall_score * 100, 1)

        except Exception as e:
            pass

    # Averages
    latin_scores = [model_data[(l, "")] for l in latin_langs if (l, "") in model_data]
    if latin_scores:
        model_data[("Latin_Avg", "")] = round(sum(latin_scores) / len(latin_scores), 1)

    non_latin_scores = [model_data[(l, "")] for l in non_latin_langs if (l, "") in model_data]
    if non_latin_scores:
        model_data[("Non-Latin_Avg", "")] = round(sum(non_latin_scores) / len(non_latin_scores), 1)

    # Output Printing
    print("[Latin]")
    for l in latin_langs:
        val = model_data.get((l, ""), "N/A")
        print(f"  {l}: {val}")
    print(f"  Avg: {model_data.get(('Latin_Avg', ''), 'N/A')}")
    
    print("\n[Non-Latin]")
    for l in non_latin_langs:
        val = model_data.get((l, ""), "N/A")
        print(f"  {l}: {val}")
    print(f"  Avg: {model_data.get(('Non-Latin_Avg', ''), 'N/A')}")

    print("\n[Overall]")
    print(f"  All   : {model_data.get(('Overall', 'All'), 'N/A')}")
    print(f"  Digit.: {model_data.get(('Overall', 'Digit.'), 'N/A')}")
    print(f"  Photo.: {model_data.get(('Overall', 'Photo.'), 'N/A')}")
    print("\n")

if __name__ == "__main__":
    main()
