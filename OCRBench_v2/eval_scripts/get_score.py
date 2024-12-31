import os
import json
import ipdb
import argparse


def calculate_average(scores_dict):
    averages = {key: sum(values) / len(values) for key, values in scores_dict.items() if len(values) > 0}
    return averages


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a JSON file to calculate scores.")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the JSON file containing inference data.")
    args = parser.parse_args()

    # Load data from JSON file
    inference_file = args.json_file
    if not os.path.exists(inference_file):
        print(f"Error: File '{inference_file}' does not exist.")
        return

    with open(inference_file, "r") as f:
        data_list = json.load(f)

    en_text_recognition_list, en_text_detection_list, en_text_spotting_list, en_relationship_extraction_list = [], [], [], []
    en_element_parsing_list, en_mathematical_calculation_list, en_visual_text_understanding_list = [], [], []
    en_knowledge_reasoning_list = []

    cn_text_recognition_list, cn_relationship_extraction_list = [], []
    cn_element_parsing_list, cn_visual_text_understanding_list = [], []
    cn_knowledge_reasoning_list = []

    res_list = []
    for item in data_list:
        if "ignore" in item.keys():
            assert item["ignore"] == "True"
        
        elif item["type"] == "text recognition en" or item["type"] == "fine-grained text recognition en" or item["type"] == "full-page OCR en":
            en_text_recognition_list.append(item["score"])
        
        elif item["type"] == "text grounding en" or item["type"] == "VQA with position en":
            en_text_detection_list.append(item["score"])
        
        elif item["type"] == "text spotting en":
            en_text_spotting_list.append(item["score"])

        elif item["type"] == "key information extraction en" or item["type"] == "key information mapping en":
            en_relationship_extraction_list.append(item["score"])
        
        elif item["type"] == "document parsing en" or item["type"] == "chart parsing en" \
        or item["type"] == "table parsing en" or item["type"] == "formula recognition en":
            en_element_parsing_list.append(item["score"])
        
        elif item["type"] == "math QA en" or item["type"] == "text counting en":
            en_mathematical_calculation_list.append(item["score"])
        
        elif item["type"] == "document classification en" \
        or item["type"] == "cognition VQA en" or item["type"] == "diagram QA en":
            en_visual_text_understanding_list.append(item["score"])
        
        elif item["type"] == "reasoning VQA en" or item["type"] == "science QA en" \
        or item["type"] == "APP agent en" or item["type"] == "ASCII art classification en":
            en_knowledge_reasoning_list.append(item["score"])
        
        elif item["type"] == "full-page OCR cn":
            cn_text_recognition_list.append(item["score"])

        elif item["type"] == "key information extraction cn" or item["type"] == "handwritten answer extraction cn":
            cn_relationship_extraction_list.append(item["score"])
        
        elif item["type"] == "document parsing cn" or item["type"] == "table parsing cn" or item["type"] == "formula recognition cn":
            cn_element_parsing_list.append(item["score"])
        
        elif item["type"] == "cognition VQA cn":
            cn_visual_text_understanding_list.append(item["score"])
        
        elif item["type"] == "reasoning VQA cn" or item["type"] == "text translation cn":
            cn_knowledge_reasoning_list.append(item["score"])

        else:
            raise ValueError("Unknown task type!")

    en_scores = {
        "text_recognition": en_text_recognition_list,
        "text_detection": en_text_detection_list,
        "text_spotting": en_text_spotting_list,
        "relationship_extraction": en_relationship_extraction_list,
        "element_parsing": en_element_parsing_list,
        "mathematical_calculation": en_mathematical_calculation_list,
        "visual_text_understanding": en_visual_text_understanding_list,
        "knowledge_reasoning": en_knowledge_reasoning_list
    }

    cn_scores = {
        "text_recognition": cn_text_recognition_list,
        "relationship_extraction": cn_relationship_extraction_list,
        "element_parsing": cn_element_parsing_list,
        "visual_text_understanding": cn_visual_text_understanding_list,
        "knowledge_reasoning": cn_knowledge_reasoning_list
    }

    en_averages = calculate_average(en_scores)
    cn_averages = calculate_average(cn_scores)

    print("English Scores:")
    for key, score in en_averages.items():
        print(f"{key}: {score:.3f} (Count: {len(en_scores[key])})")

    print("\nChinese Scores:")
    for key, score in cn_averages.items():
        print(f"{key}: {score:.3f} (Count: {len(cn_scores[key])})")

    score_en_overall = sum(en_averages.values()) / len(en_averages)
    score_cn_overall = sum(cn_averages.values()) / len(cn_averages)

    print("\nOverall Scores:")
    print(f"English Overall Score: {score_en_overall:.3f}")
    print(f"Chinese Overall Score: {score_cn_overall:.3f}")

    print("End of Code!")

if __name__ == "__main__":
    main()
