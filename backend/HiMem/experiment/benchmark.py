# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
import argparse
import json
import os
import re
import string
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process, Pool

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

sys.path.append('.')
from metrics import calculate_metrics, calculate_bleu_scores, evaluate_llm_judge

load_dotenv()
client = OpenAI()


def create_batches(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def get_answer(ans):
    strip_word_list = [
        "\nDialogs:",
        "\n[bot]:",
        "\nAssistant:",
        "\nReview:",
        "\n",
        "[bot]:",
    ]
    cut_word_list = ["\n[human]:", "\nQuestion:", "\nQ:"]

    for strip_word in strip_word_list:
        ans = ans.strip(strip_word)
    for cut_word in cut_word_list:
        if cut_word in ans:
            ans = ans.split(cut_word)[0]
    return ans


def process_item(samples):
    results = []

    for sample in tqdm(samples):
        question = sample["question"]
        gt_answer = sample["answer"]
        pred_answer = sample["prediction"]
        search_latency = sample["search_latency"]
        total_latency = sample["total_latency"]
        total_tokens = sample["total_tokens"]
        memories = sample["retrieved_memories"]

        metrics = calculate_metrics(pred_answer, gt_answer)
        llm_score = evaluate_llm_judge(client, question, gt_answer, pred_answer)
        record = {
            "question": question,
            "category": sample["category"],
            "accuracy": llm_score,
            "answer": gt_answer,
            "prediction": pred_answer,
            "search_latency": search_latency,
            "total_latency": total_latency,
            "total_tokens": total_tokens,
            "retrieved_memories": memories,
        }
        record.update(metrics)
        results.append(record)

    return results


def eval_locomo(save_path):
    # -------------------------------
    # 1. Load the JSONL file
    # -------------------------------
    # Replace 'your_results.jsonl' with your actual file path
    df = pd.read_json(save_path, lines=True)

    # Optional: quick look at the data
    print("\nTotal number of examples:", len(df))

    locomo_category_mappings = {4: 'single-hop', 1: 'multi-hop', 2: 'temporal-reasoning', 3: 'open-domain', }
    category_order = [4, 1, 2, 3]
    metric_columns = [
        'accuracy', 'f1',
        'bleu1',  # 'bleu2', 'bleu3', 'bleu4',
        # 'rouge1_f', 'rouge2_f', 'rougeL_f', 'exact_match'
        # 'bert_precision', 'bert_recall', 'bert_f1',
        'search_latency', 'total_latency', 'total_tokens'
    ]
    metric_columns = [col for col in metric_columns if col in df.columns]

    overall_means = df[metric_columns].mean()
    print("=== OVERALL MEANS ===")
    print(overall_means.round(4))

    # -------------------------------
    # 5. Means grouped by category → with text names & nice order
    # -------------------------------
    if 'category' in df.columns:
        grouped_means = (
            df.groupby('category')[metric_columns]
            .mean()
            .round(4)
            .reindex(category_order)  # enforce desired row order
            .rename(index=locomo_category_mappings)  # replace numbers with text
            .dropna(how='all')  # remove rows if a whole category is missing
        )
        grouped_means.index.name = 'Category'

        print("\n=== MEANS GROUPED BY CATEGORY ===")
        print(grouped_means)

        # Counts with the same order and naming
        counts = (
            df['category']
            .value_counts()
            .reindex(category_order, fill_value=0)
            .rename(locomo_category_mappings)
        )

        print("\nNumber of examples per category:")
        print(counts.to_string())
    else:
        print("No 'category' column found.")
        raise Exception("No 'category' column found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="long-term conversation evaluation")
    parser.add_argument("--dataset", type=str, default="locomo",
                        help="Name of dataset")
    parser.add_argument('--env', type=str, default='dev')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="OpenAI model to use")
    parser.add_argument('--search_mode', type=str, default=None,
                        help="Search mode, hybrid, best-effort, note or episode")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Num of fetched results (5,10,15,20,25)")
    parser.add_argument("--round", type=int, default=None, )
    parser.add_argument(
        "--load_path", default=None
    )
    parser.add_argument(
        "--save_path", default=None
    )
    parser.add_argument(
        "--force_overwrite", default=False, action="store_true"
    )
    args = parser.parse_args()
    load_path = args.load_path
    save_path = args.save_path
    name = args.name
    if not name or name == 'None':
        name = f"{args.dataset}"
    else:
        name = f"{args.dataset}/{name}"
    if not load_path:
        if args.search_mode:
            load_path = f"output/{name}/generation_results_{args.model}_{args.env}_{args.search_mode}_{args.top_k}"
        else:
            load_path = f"output/{name}/generation_results_{args.model}_{args.env}_{args.top_k}"
    if not save_path:
        if args.search_mode:
            save_path = f"output/{name}/evaluation_results_{args.model}_{args.env}_{args.search_mode}_{args.top_k}"
        else:
            save_path = f"output/{name}/evaluation_results_{args.model}_{args.env}_{args.top_k}"
    if args.round:
        load_path += f"_{args.round}"
        save_path += f"_{args.round}"
    load_path += ".jsonl"
    save_path += ".jsonl"
    if not os.path.exists(load_path):
        print(f"No such file: {load_path}")
        exit(1)
    is_exists = os.path.exists(save_path)
    if not is_exists or (is_exists and args.force_overwrite):
        samples = []
        with open(load_path, "r") as f:
            for line in f.readlines():
                sample = json.loads(line.strip())
                samples.append(sample)

        batches = create_batches(samples, 76)
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_item, batch) for batch in batches]
            for future in as_completed(futures):
                results.extend(future.result())
        # Save results to JSON file
        with open(save_path, "w", encoding="utf-8") as f:
            f.writelines([json.dumps(_, ensure_ascii=False) + "\n" for _ in results])

    # Load the evaluation metrics data
    data = []
    with open(save_path, "r") as f:
        for line in f.readlines():
            sample = json.loads(line.strip())
            data.append(sample)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Calculate mean scores by category
    result = df.agg(
        {"accuracy": "mean", "f1": "mean", "rougeL_f": "mean",
         "exact_match": "mean",
         "bleu1": "mean", "bleu2": "mean", "bleu3": "mean", "bleu4": "mean",
         "search_latency": "mean", "total_latency": "mean", "total_tokens": "mean"}).round(4)
    print("\nOverall Mean Scores:")
    print(result)

    eval_locomo(save_path)
