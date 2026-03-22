import argparse
import os
import sys
import json
import time
import glob
from collections import defaultdict
import uuid

# Add HiMem to module path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(backend_dir, 'HiMem'))
sys.path = [p for p in sys.path if 'locomo/.venv' not in p]
# Allow GPU usage by not setting CUDA_VISIBLE_DEVICES to empty
os.chdir(os.path.join(backend_dir, 'HiMem'))

try:
    from openai import OpenAI
    import yaml
    from himem.configs.base import MemoryConfig
    from himem.memory.main import Memory
    from himem.dataset.model import Session, Turn
    from experiment.metrics import evaluate_llm_judge
except ImportError as e:
    print(f"Failed to import HiMem dependencies. Make sure you have installed them. Error: {e}")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(backend_dir, '.env'))
except ImportError:
    env_path = os.path.join(backend_dir, '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip() and not line.strip().startswith('#'):
                    parts = line.strip().split('=', 1)
                    if len(parts) == 2:
                        os.environ[parts[0].strip()] = parts[1].strip("'\"")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_summary(previous_summary, new_dialogue):
    if not previous_summary:
        prompt = f"Dialogue:\n{new_dialogue}\n\nSummarize the conversation comprehensively."
    else:
        prompt = f"Previous Summary:\n{previous_summary}\n\nNew Dialogue:\n{new_dialogue}\n\nBriefly update the summary with the new dialogue. Return the updated comprehensive summary. Ensure NO important facts are lost."
    
    retries = 3
    while retries > 0:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            retries -=1
            time.sleep(2)
    return previous_summary

def answer_with_context(context, question):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nBased on the context, answer the question in a short phrase. If not mentioned in the context, explicitly state 'Not mentioned in the text.'\nAnswer:"
    retries = 3
    while retries > 0:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            retries -= 1
            time.sleep(2)
    return "Error generating answer"

def format_session_dialogue(session_data):
    lines = []
    for d in session_data:
        speaker = d.get('speaker', 'Unknown')
        text = d.get('clean_text', d.get('text', ''))
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)

def init_himem():
    config_path = os.path.join(backend_dir, 'HiMem', 'config', 'base.yaml')
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    config_dict["llm_providers"]["openai"]["config"]["api_key"] = os.environ.get("OPENAI_API_KEY", "")
    config_dict["llm_providers"]["openai"]["config"]["model"] = "gpt-4o-mini"
    
    # Use smaller models that fit on a 4GB GPU
    config_dict["embedder"]["config"]["model"] = "all-MiniLM-L6-v2"
    config_dict["embedder"]["config"]["embedding_dims"] = 384
    if "vector_store" in config_dict and "config" in config_dict["vector_store"]:
        config_dict["vector_store"]["config"]["embedding_model_dims"] = 384
    if "reranker" in config_dict and "config" in config_dict["reranker"]:
        config_dict["reranker"]["config"]["model"] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Ensure memory collection is uniquely named for this run to avoid collisions
    run_id = str(uuid.uuid4())[:8]
    if 'vector_store' in config_dict and 'config' in config_dict['vector_store']:
        config_dict['vector_store']['config']['collection_name'] = f"benchmark_{run_id}"
    if 'components' in config_dict and 'segmentor' in config_dict['components'] and 'config' in config_dict['components']['segmentor']:
        config_dict['components']['segmentor']['config']['collection_name'] = f"topics_{run_id}"
    if 'components' in config_dict and 'episode_memory' in config_dict['components'] and 'config' in config_dict['components']['episode_memory']:
        config_dict['components']['episode_memory']['config']['index_prefix'] = f"benchmark_{run_id}"
    config = MemoryConfig(**config_dict)
    memory = Memory(config)
    return memory

def process_dataset_ira(limit=None):
    base_dir = os.path.join(backend_dir, 'locomo', 'data', 'multimodal_dialog', 'ira_long')
    conv_folders = sorted(glob.glob(os.path.join(base_dir, 'ira-conv-*')))
    if limit:
        conv_folders = conv_folders[:limit]

    results = []
    
    print(f"Loading {len(conv_folders)} IRA conversations...", flush=True)
    himem_memory = init_himem()
    
    for folder in conv_folders:
        agent_a_path = os.path.join(folder, 'agent_a.json')
        q_path = os.path.join(folder, 'questions.json')
        if not os.path.exists(agent_a_path) or not os.path.exists(q_path):
            continue
            
        data = read_json(agent_a_path)
        questions_doc = read_json(q_path)
        
        user_id_himem = f"user_{os.path.basename(folder)}"
        
        # 1. Gather all sessions
        sessions = []
        i = 1
        while f"session_{i}" in data:
            sessions.append(data[f"session_{i}"])
            i += 1
            
        # 2. Build Memory (Baseline and HiMem)
        summary = ""
        print(f"Processing IRA: {os.path.basename(folder)} with {len(sessions)} sessions...", flush=True)
        for idx, sess in enumerate(sessions):
            dialogue = format_session_dialogue(sess)
            summary = generate_summary(summary, dialogue)
            
            # Format as requested by HiMem
            # HiMem takes a list of dicts with role and content.
            # Convert our speakers to user/assistant.
            himem_turns = []
            for t_idx, d in enumerate(sess):
                speaker = d.get('speaker', '')
                text = d.get('clean_text', d.get('text', ''))
                role = "user" if speaker.lower() != "ira" else "assistant"
                himem_turns.append(Turn(dia_id=str(t_idx), role=role, content=text))
            
            if himem_turns:
                session_obj = Session(session_id=str(idx), date_time="2024-01-01 10:00:00", turns=himem_turns)
                himem_memory.add(session=session_obj, user_id=user_id_himem, construction_mode='all')
                
        # 3. Answer questions
        for cat in questions_doc.get("categories", []):
            cat_name = cat["category_name"]
            for q_obj in cat.get("questions", []):
                question = q_obj["question"]
                gt = q_obj["ground_truth"]
                
                # Baseline
                baseline_ans = answer_with_context(summary, question)
                baseline_score = evaluate_llm_judge(client, question, gt, baseline_ans)
                
                # HiMem
                retrieved = himem_memory.search(query=question, user_id=user_id_himem, limit=10, mode='hybrid')
                if isinstance(retrieved, dict):
                    retrieved = retrieved.get('results', [])
                retrieved_context = "\n".join([r.get('content', '') if isinstance(r, dict) else str(r) for r in retrieved])
                himem_ans = answer_with_context(retrieved_context, question)
                himem_score = evaluate_llm_judge(client, question, gt, himem_ans)
                
                results.append({
                    "dataset": "ira_long",
                    "category": cat_name,
                    "question": question,
                    "ground_truth": gt,
                    "baseline_context": summary,
                    "baseline_ans": baseline_ans,
                    "baseline_score": baseline_score,
                    "himem_context": retrieved_context,
                    "himem_ans": himem_ans,
                    "himem_score": himem_score
                })
                
    return results



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ira-limit", type=int, default=10, help="Number of IRA convos to test")
    parser.add_argument("--out", type=str, default="benchmark_results_ira.md")
    parser.add_argument("--json-out", type=str, default="benchmark_details_ira.json")
    args = parser.parse_args()
    
    all_results = []
    print("Starting Benchmark...", flush=True)
    all_results.extend(process_dataset_ira(limit=args.ira_limit))
    
    # Aggregate results
    agg = defaultdict(lambda: defaultdict(lambda: {"baseline": [], "himem": []}))
    for r in all_results:
        ds = r['dataset']
        cat = r['category']
        agg[ds][cat]["baseline"].append(r['baseline_score'])
        agg[ds][cat]["himem"].append(r['himem_score'])
        
    # Write MarkDown
    out_path = os.path.join(backend_dir, args.out)
    with open(out_path, 'w') as f:
        f.write("# Benchmark Results: HiMem vs Baseline (Running Summary)\n\n")
        f.write("This report details accuracy scores scaled 0-1, evaluated by GPT-4 as a judge.\n\n")
        
        for ds, cat_dict in agg.items():
            f.write(f"## Dataset: {ds}\n\n")
            f.write("| Category | Baseline Avg Score | HiMem Avg Score | Count |\n")
            f.write("|----------|--------------------|-----------------|-------|\n")
            
            ds_b_all = []
            ds_h_all = []
            for cat, scores in cat_dict.items():
                b_avg = sum(scores["baseline"]) / max(len(scores["baseline"]), 1)
                h_avg = sum(scores["himem"]) / max(len(scores["himem"]), 1)
                count = len(scores["baseline"])
                ds_b_all.extend(scores["baseline"])
                ds_h_all.extend(scores["himem"])
                f.write(f"| {cat} | {b_avg:.3f} | {h_avg:.3f} | {count} |\n")
                
            tot_b = sum(ds_b_all) / max(len(ds_b_all), 1)
            tot_h = sum(ds_h_all) / max(len(ds_h_all), 1)
            f.write(f"| **Overall** | **{tot_b:.3f}** | **{tot_h:.3f}** | **{len(ds_b_all)}** |\n\n")
            
    # Write JSON details
    json_path = os.path.join(backend_dir, args.json_out)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
                
    print(f"Done! Results written to {out_path} and {json_path}")

if __name__ == "__main__":
    main()
