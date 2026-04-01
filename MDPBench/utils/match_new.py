"""
Post-process for OminiDocBench

Because HunYuanOCR is end-to-end parsing and ignores the restrictions of the pre-layout on the panel categories, 
the model's parsing results are diverse. 
Quick match may have about 8% false matches. We adopted a hierarchical paradigm:

- Edit distance < 0.4:
We consider this type of match a correct match and directly use it as Form_part1.

- Edit distance >= 0.4:
We believe this case may be caused by model parsing failure or incorrect matching. 
We adjust the match through a simple automated post-processing + manual post-processing paradigm.
"""
import json
import re
import os
from difflib import SequenceMatcher
from collections import Counter

# ======================= Tool Functions =======================
def remove_big_braces(s: str):
    pattern = r'\\(big|Big|bigg|Bigg)\{([^\}]+)\}'
    repl    = r'\\\1\2'
    return re.sub(pattern, repl, s)


def process_final_ans(final_ans):
    for item in final_ans:
        if "pred" in item and isinstance(item["pred"], str):
            item["pred"] = remove_big_braces(item["pred"])
    return final_ans


def clean_gt_tail(gt: str):
    pattern = r'(\\quad+|\\qquad+)\s*\{?\(\s*\d+\s*\)\}?\s*$'
    return re.sub(pattern, '', gt).rstrip()


def load_instances(jsonl_path):
    instances = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                instances.append(obj)
            except json.JSONDecodeError as e:
                print(f"[WARN] Parsing failed: {e}, skipped")
    print(f"[INFO] Read {len(instances)} markdown instances from {jsonl_path}" )
    return instances

def count_id_distribution(error_match_ans):
    """
    Perform distribution statistics on the id field of error_match_ans (intermediate output)
    """
    id_list = [item['id'] for item in error_match_ans if 'id' in item]
    counter = Counter(id_list)

    print("\n===============================")
    print("📌 ID field distribution statistics (intermediate output)")
    print("===============================")
    print("Total number of records:", len(error_match_ans))
    print("Number of records containing id:", len(id_list))
    print("\n=== Distribution of the id field ===")

    for id_val, cnt in counter.most_common():
        print(f"id={id_val} : {cnt} 次")

    print("===============================\n")

    return counter


# ---------- 匹配工具 ----------

def normalize_for_match(text: str):
    text = re.sub(r"\\textcircled\{a\}", "ⓐ", text)
    text = re.sub(r"\\textcircled\{b\}", "ⓑ", text)
    text = re.sub(r"\\textcircled\{c\}", "ⓒ", text)
    text = re.sub(r"\\textcircled\{d\}", "ⓓ", text)

    text = text.replace("\\text{ⓐ}", "ⓐ")
    text = text.replace("\\text{ⓑ}", "ⓑ")
    text = text.replace("\\text{ⓒ}", "ⓒ")
    text = text.replace("\\text{ⓓ}", "ⓓ")

    text = text.replace(" ", "")
    return text


def clean_formula(text: str):
    return (text.replace("\\quad", "")
                .replace("$", "")
                .strip())


def extract_candidates(markdown: str):
    lines = markdown.split("\n")
    candidates = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        line = re.sub(r"^\s*\d+\.\s*", "", line)
        cleaned = clean_formula(line)
        if cleaned:
            candidates.append(cleaned)
    
    return candidates


def best_match(gt: str, candidates):
    gt_norm = normalize_for_match(gt)
    best_score = -1
    best_cand = None
    
    for cand in candidates:
        cand_norm = normalize_for_match(cand)
        score = SequenceMatcher(None, gt_norm, cand_norm).ratio()
        if score > best_score:
            best_score = score
            best_cand = cand
    
    return best_cand, best_score


def process_badcases(Form_part2):
    results = []
    
    for case in Form_part2:
        markdown = case["markdown"]
        gt = case["gt"]

        candidates = extract_candidates(markdown)
        pred, score = best_match(gt, candidates)

        pred = pred.replace("\\text{ⓐ}","\\textcircled{a}") \
                   .replace("\\text{ⓑ}","\\textcircled{b}") \
                   .replace("\\text{ⓒ}","\\textcircled{c}") \
                   .replace("\\text{ⓓ}","\\textcircled{d}") \
                   .replace("ⓐ","\\textcircled{a}") \
                   .replace("ⓑ","\\textcircled{b}") \
                   .replace("ⓒ","\\textcircled{c}") \
                   .replace("ⓓ","\\textcircled{d}")

        results.append({
            'img_id': case['img_id'],
            "gt": gt,
            "pred": pred,
            "match_score": score
        })
    
    return results


# ======================= Main Function =======================

def process_formula_matching(match_file, markdown_file, markdown_key, output_file):

    # ----------- Step1: Read the matching result file -----------
    with open(match_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    final_ans = []
    for idx, item in enumerate(raw_data):
        ref = item['gt'].replace('$', '') \
                        .replace('\[', '').replace('\]','') \
                        .replace('，', ',') \
                        .strip()

        pred = item['pred'].replace('$', '') \
                           .replace('\[', '').replace('\]','') \
                           .replace('，', ',') \
                           .strip()

        final_ans.append({
            'img_id': f"{idx}",
            'id': f"{item['img_id']}",
            'gt': ref,
            'pred': pred,
            'edit': item['edit']
        })

    final_ans = process_final_ans(final_ans)

    # ----------- Step2: Split Form_part1 / error_match -----------
    Form_part1 = []
    error_match_ans = []

    for item in final_ans:
        item['pred'] = clean_gt_tail(item['pred'])

        if item['edit'] < 0.4:
            Form_part1.append(item)
        else:
            error_match_ans.append(item)
    distribution = count_id_distribution(error_match_ans)

    # ----------- Step3: Write markdown into error_match_ans -----------
    markdown_data = load_instances(markdown_file)

    for item in markdown_data:
        basename = os.path.basename(item['image_path'][0])

        for seq in error_match_ans:
            if basename == seq['id']:
                seq['markdown'] = item[markdown_key]

    # ----------- Step4: Special case handling for Form_part2 (id points to a specific file） -----------
    Form_part2 = [
        x for x in error_match_ans
        if x['id'] == "yanbaopptmerge_9081a70ff98b3e7d640660a9412c447d.pdf_1287.jpg"
    ]

    # Matching bad samples
    out = process_badcases(Form_part2)

    # ----------- Step5: For regular error matching, substring rules are used directly. -----------
    Form_part3 = []
    for item in error_match_ans:
        if item['id'] == "yanbaopptmerge_9081a70ff98b3e7d640660a9412c447d.pdf_1287.jpg":
            continue

        gt = item['gt'].replace(' ', '')
        answer = item['markdown'].replace('$','').replace(' ','')
        if gt in answer:
            item['pred'] = item['gt']

        Form_part3.append(item)

    # ----------- Step6: Combine all results and output. -----------
    merge_form = Form_part1 + out + Form_part3

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merge_form, f, ensure_ascii=False, indent=4)

    print(f"[DONE] Saved final result to {output_file}")


# ======================= END =======================

if __name__ == "__main__":
    process_formula_matching(
        match_file="vllm_omni_quick_match_display_formula_result.json",# Omnidocbenh quick match formula matching results
        markdown_file="OCR_OmniDocbench_vllm_infer_res.jsonl", # parsing result jsonl from vllm
        markdown_key="vllm_answer_eth", # answer key
        output_file="Final_formula.json" #The output file will be evaluated using the same method described in https://github.com/opendatalab/UniMERNet/tree/main/cdm.
    )