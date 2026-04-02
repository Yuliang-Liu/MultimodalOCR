# from modules.cal_matrix import cal_text_matrix, cal_table_teds
from registry.registry import EVAL_TASK_REGISTRY
from metrics.show_result import show_result, get_full_labels_results, get_page_split
from registry.registry import METRIC_REGISTRY
import json
import os
import pdb

@EVAL_TASK_REGISTRY.register("end2end_eval")
class End2EndEval():
    def __init__(self, dataset, metrics_list, page_info_path, save_name):
        result_all = {}
        page_info = {}
        if not os.path.exists(f'./result/{save_name}'):
            os.makedirs(f'./result/{save_name}')

        if os.path.isdir(page_info_path):
            md_flag = True
        else:
            md_flag = False
        if not md_flag:
            with open(page_info_path, 'r') as f:
                pages = json.load(f)
            
            for page in pages:
                img_path = os.path.basename(page['page_info']['image_path'])
                page_info[img_path] = page['page_info']['page_attribute']

        import copy
        for element in metrics_list.keys():
            samples_all_obj = dataset.samples[element]
            
            # Helper to filter samples
            def get_filtered_samples(dataset_obj, is_digit_flag):
                if isinstance(dataset_obj, list):
                    raw_samples = dataset_obj
                    is_obj = False
                else:
                    raw_samples = dataset_obj.samples
                    is_obj = True
                
                filtered = [s for s in raw_samples if s.get('is_digit', True) == is_digit_flag]
                
                if is_obj:
                    new_obj = copy.copy(dataset_obj)
                    new_obj.samples = filtered
                    return new_obj
                else:
                    return filtered

            samples_digit = get_filtered_samples(samples_all_obj, True)
            samples_photo = get_filtered_samples(samples_all_obj, False)
            
            eval_groups = [
                ('All', samples_all_obj),
                ('digit', samples_digit),
                ('photo', samples_photo)
            ]
            
            result_all[element] = {}

            for group_name, group_samples in eval_groups:
                # Check if empty
                if isinstance(group_samples, list):
                    count = len(group_samples)
                else:
                    count = len(group_samples.samples)
                
                if count == 0:
                    continue

                result = {}
                group_info = metrics_list[element].get('group', [])
                
                for metric in metrics_list[element]['metric']:
                    metric_val = METRIC_REGISTRY.get(metric)
                    # evaluate returns (samples, result_dict)
                    group_samples, result_s = metric_val(group_samples).evaluate(group_info, save_name, f"{element}_{group_name}")
                    if result_s:
                        result.update(result_s)
                
                if result:
                    print(f'【{element} - {group_name}】')
                    show_result(result)
                
                if group_name == 'All':
                    key_suffix = ''
                else:
                    key_suffix = f'_{group_name}'
                
                if md_flag:
                    group_result =  {}
                    page_result = {}
                else:
                    group_result = get_full_labels_results(group_samples)
                    page_result = get_page_split(group_samples, page_info)
                
                result_all[element][f'all{key_suffix}'] = result
                result_all[element][f'group{key_suffix}'] = group_result
                result_all[element][f'page{key_suffix}'] = page_result
            
            samples = samples_all_obj # Restore for saving
            # pdb.set_trace()

            if not os.path.exists(f'./result/{save_name}'):
                os.makedirs(f'./result/{save_name}')
            if isinstance(samples, list):
                saved_samples = samples
            else:
                saved_samples = samples.samples
            try:

                with open(f'./result/{save_name}/{save_name}_{element}_result.json', 'w', encoding='utf-8') as f:
                    json.dump(saved_samples, f, indent=4, ensure_ascii=False)
            except TypeError as e:
                print(f"JSON 序列化错误: {e}")
                print("请检查 saved_samples 中是否包含非 JSON 可序列化的数据类型")
                
                # 打印出有问题的数据类型
                def find_non_serializable(data):
                    if isinstance(data, dict):
                        for k, v in data.items():
                            try:
                                json.dumps(v)
                            except TypeError:
                                print(f"键 '{k}' 包含不可序列化的值: {v} (类型: {type(v)})")
                                find_non_serializable(v)
                    elif isinstance(data, (list, tuple)):
                        for i, item in enumerate(data):
                            try:
                                json.dumps(item)
                            except TypeError:
                                print(f"索引 {i} 包含不可序列化的项: {item} (类型: {type(item)})")
                                find_non_serializable(item)
                
                find_non_serializable(saved_samples)


        with open(f'./result/{save_name}/{save_name}_metric_result.json', 'w', encoding='utf-8') as f:
            json.dump(result_all, f, indent=4, ensure_ascii=False)
    