import sys, os, json
sys.path.append('/home/zhangli/miniconda3/envs/omnidocbench/OmniDocBench')
from dataset.end2end_dataset import End2EndDataset
p_json='/home/zhangli/miniconda3/envs/omnidocbench/OmniDocBench/demo_data/omnidocbench_demo/OmniDocBench_demo.json'
pred_folder='/home/zhangli/miniconda3/envs/omnidocbench/OmniDocBench/demo_data/omnidocbench_demo/mds'
cfg={'dataset':{'ground_truth':{'data_path':p_json},'prediction':{'data_path':pred_folder},'match_method':'quick_match'}}

print('Instantiating End2EndDataset...')
d = End2EndDataset(cfg)
print('Done. Collecting samples...')
samples = d.samples
out = {}
for k in ['display_formula','text_block','table','reading_order']:
    ds = samples.get(k)
    if not ds:
        out[k] = []
        continue
    try:
        s = ds.samples
    except Exception:
        s = ds
    # normalize to python native types
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {make_serializable(k): make_serializable(v) for k,v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        if isinstance(obj, (int, float, str, type(None), bool)):
            return obj
        try:
            return obj.tolist()
        except Exception:
            return str(obj)
    out[k] = [make_serializable(item) for item in s]

out_path = '/home/zhangli/miniconda3/envs/omnidocbench/OmniDocBench/demo_data/omnidocbench_demo/quick_match_results.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print('Saved quick_match results to:', out_path)
for k in out:
    print(k, 'count =', len(out[k]))