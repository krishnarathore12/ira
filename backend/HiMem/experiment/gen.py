import argparse
import json
import os
import sys
import time

import yaml
from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm

sys.path.append('.')
from experiment.prompts import ANSWER_PROMPT, ANSWER_PROMPT_EXTRA
from himem.configs.base import MemoryConfig
from himem.memory.main import Memory

load_dotenv()


class MemorySearch:
    def __init__(self, config, search_mode: str = 'knowledge', top_k=10):
        self.memory = Memory(config)
        self.openai_client = OpenAI()
        self.model = "gpt-4o-mini"
        self.top_k = top_k
        self.search_mode = search_mode

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0
        memories = []
        while retries < max_retries:
            try:
                print(f"query: {query}, user: {user_id}")
                memories = self.memory.search(
                    query,
                    user_id=user_id,
                    limit=self.top_k,
                    mode=self.search_mode,
                )
                if isinstance(memories, dict):
                    memories = memories.get('results', [])
                break
            except Exception as e:
                retries += 1
                print(f"Retrying...{retries}")
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)

        end_time = time.time()
        semantic_memories = [
            {
                "memory": memory["content"],
                "timestamp": memory["metadata"].get("timestamp", ""),
            }
            for memory in memories
        ]
        return semantic_memories, end_time - start_time

    def answer_question_of_locomo(self, speakers, question):
        start_time = time.time()
        memories = []
        if len(speakers) == 1:
            speaker = speakers[0]
            speaker_memories, speaker_memory_time = self.search_memory(
                speaker, question
            )
            speaker_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_memories]
            template = Template(ANSWER_PROMPT)
            answer_prompt = template.render(
                speaker_here=speaker.split("_")[0],
                memories_here=json.dumps(speaker_memory, indent=4),
                question=question,
            )
            memories.extend(speaker_memory)
        else:
            speaker_1 = speakers[0]
            speaker_2 = speakers[1]
            speaker_1_memories, speaker_1_memory_time = self.search_memory(
                speaker_1, question
            )
            speaker_2_memories, speaker_2_memory_time = self.search_memory(
                speaker_2, question
            )
            search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
            search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

            template = Template(ANSWER_PROMPT_EXTRA)
            answer_prompt = template.render(
                speaker_1_user_id=speaker_1.split("_")[0],
                speaker_2_user_id=speaker_2.split("_")[0],
                speaker_1_memories=json.dumps(search_1_memory, indent=4),
                speaker_2_memories=json.dumps(search_2_memory, indent=4),
                question=question,
            )
            memories.extend(search_1_memory)
            memories.extend(search_2_memory)
        search_latency = time.time() - start_time
        retries = 0

        while retries < 3:
            response = self.openai_client.chat.completions.create(
                model=self.model, messages=[{"role": "system", "content": answer_prompt}], temperature=0.0
            )
            answer = response.choices[0].message.content.strip()
            if len(answer) == 0 or answer == 'Answer:':
                retries += 1
            else:
                break
        total_latency = time.time() - start_time
        return (
            response.choices[0].message.content,
            response.usage.total_tokens,
            search_latency,
            total_latency,
            memories
        )

    def answer_question_of_longmemeval(self, speaker, question):
        start_time = time.time()
        memories, memory_time = self.search_memory(
            speaker, question
        )
        search_latency = time.time() - start_time

        template = Template(ANSWER_PROMPT)
        answer_prompt = template.render(
            memories_here=json.dumps(memories, indent=4),
            question=question,
        )

        response = self.openai_client.chat.completions.create(
            model=self.model, messages=[{"role": "system", "content": answer_prompt}], temperature=0.0
        )
        total_latency = time.time() - start_time
        return (
            response.choices[0].message.content,
            response.usage.total_tokens,
            search_latency,
            total_latency,
            memories
        )

    def process_question_of_locomo(self, val, speakers):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        if not answer:
            answer = val.get("adversarial_answer", "")

        predicted_answer, total_tokens, search_latency, total_latency, memories = self.answer_question_of_locomo(speakers,
                                                                                                                 question)
        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "prediction": predicted_answer,
            "total_tokens": total_tokens,
            "search_latency": search_latency,
            "total_latency": total_latency,
            "retrieved_memories": memories,
        }

        return result

    def process_conversation_of_locomo(self, idx, item, processed_questions, force_overwrite) -> []:
        results = []
        original_qas = item["qa"]
        conversation = item["conversation"]

        qas = []
        for qa in original_qas:
            category = int(qa["category"])
            if category == 5:
                continue
            qas.append(qa)

        for question_item in tqdm(
                qas, total=len(qas), desc=f"Processing questions for conversation {idx}", leave=False
        ):
            question = question_item["question"]
            if question in processed_questions:
                continue
            speakers = self.fetch_speakers_from_question(question, conversation)
            result = self.process_question_of_locomo(question_item, speakers)

            results.append(result)
        return results

    def fetch_speakers_from_question(self, question, conversation) -> []:
        speakers = []
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        if speaker_a in question and speaker_b in question:
            speakers.append(speaker_a)
            speakers.append(speaker_b)
        elif speaker_a in question:
            speakers.append(speaker_a)
        elif speaker_b in question:
            speakers.append(speaker_b)
        else:
            speakers.append(speaker_a)
            speakers.append(speaker_b)
        return speakers

    def process_conversation_of_longmemeval(self, idx, item, processed_questions, force_overwrite) -> []:
        question = item["question"]
        category = item["question_type"]
        answer = item["answer"]
        speaker = 'xiaowu'

        if question in processed_questions and not force_overwrite:
            return []
        predicted_answer, total_tokens, search_latency, total_latency, _ = self.answer_question_of_longmemeval(speaker, question)
        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "prediction": predicted_answer,
            "total_tokens": total_tokens,
            "search_latency": search_latency,
            "total_latency": total_latency,
        }

        return [result]

    def process_data_file(self, dataset_name, env, save_path, ratio: float = 1.0, force_overwrite=False):
        results = []

        dataset_path = f"data/{dataset_name}_{env}.json"
        with open(dataset_path, "r") as f:
            data = json.load(f)

        data = data[:int(ratio * len(data))]

        # Load processed questions
        processed_questions = set()
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                for line in f.readlines():
                    sample = json.loads(line.strip())
                    processed_questions.add(sample["question"])
                    results.append(sample)
        print(f"have processed {len(processed_questions)} questions, {len(results)} samples")
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            if dataset_name == 'locomo':
                items = self.process_conversation_of_locomo(idx, item, processed_questions, force_overwrite)
            else:
                items = self.process_conversation_of_longmemeval(idx, item, processed_questions, force_overwrite)
            results.extend(items)
            with open(save_path, "w", encoding="utf-8") as f:
                f.writelines([json.dumps(_, ensure_ascii=False) + "\n" for _ in results])
        print(f"Processed total {len(results)} questions")


def main():
    parser = argparse.ArgumentParser(description="Evaluate text-only agent on LoComo dataset")
    parser.add_argument("--dataset", type=str, default="locomo",
                        help="Name of dataset")
    parser.add_argument('--env', type=str, default='dev')
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="OpenAI model to use")
    parser.add_argument('--search_mode', type=str, default='hybrid',
                        help="Search mode, hybrid, best-effort, note or episode")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Num of fetched results (5,10,15,20,25)")
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument("--round", type=int, default=None, )
    parser.add_argument("--force_overwrite", default=False, action="store_true")
    args = parser.parse_args()

    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")

    check_point_file_path = f"data/{args.dataset}_{args.env}_memory_construction_processing_checkpoint.pkl"
    if not os.path.exists(check_point_file_path):
        print(f"Checkpoint file {check_point_file_path} does not exist")
        exit(1)

    save_path = args.save_path
    name = args.name
    if not name or name == 'None':
        name = f"{args.dataset}"
    else:
        name = f"{args.dataset}/{name}"
    if not save_path:
        if args.round:
            save_path = f"output/{name}/generation_results_{args.model}_{args.env}_{args.search_mode}_{args.top_k}_{args.round}.jsonl"
        else:
            save_path = f"output/{name}/generation_results_{args.model}_{args.env}_{args.search_mode}_{args.top_k}.jsonl"

    is_exists = os.path.exists(save_path)

    if not is_exists or (is_exists and args.force_overwrite):
        config_file_path = f"config/{args.env}.yaml"
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Config file not found: {config_file_path}")
        with open(config_file_path, 'r') as stream:
            config = yaml.safe_load(stream)
        config = MemoryConfig(**config)
        handler = MemorySearch(config, args.search_mode, args.top_k)
        handler.process_data_file(args.dataset, args.env, save_path, args.ratio, args.force_overwrite)


if __name__ == "__main__":
    main()
