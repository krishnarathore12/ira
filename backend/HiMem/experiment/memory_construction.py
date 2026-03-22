import argparse
import json
import os
import sys

import yaml
from loguru import logger
from tqdm import tqdm

sys.path.append('.')
from experiment.checkpoint_manager import ProcessingStateManager
from himem.configs.base import MemoryConfig
from himem.dataset.locomo_dataset_loader import load_locomo_dataset
from himem.dataset.longmemeval_dataset_loader import load_longmemeval_dataset
from himem.memory.main import Memory


def load_data(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    return data


def build_memory_index(dataset_name: str, env: str, ratio: float = 1, construction_mode='all', enable_knowledge_alignment=True):
    config_file_path = f"config/{env}.yaml"
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found: {config_file_path}")
    with open(config_file_path, 'r') as stream:
        config = yaml.safe_load(stream)
    config = MemoryConfig(**config)

    memory = Memory(config)

    dataset_file_path = f"data/{dataset_name}_{env}.json"
    if dataset_name == 'locomo':
        samples = load_locomo_dataset(dataset_file_path, ratio)
    else:
        samples = load_longmemeval_dataset(dataset_file_path, ratio)

    all_units = []
    for sample in samples:
        for cid, conversation in enumerate(sample.conversations):
            for session_id, session in conversation.sessions.items():
                key = f"conversation_{cid}_{conversation.user}_session_{session_id}"
                all_units.append(key)

    checkpoint_manager = ProcessingStateManager(f"data/{dataset_name}_{env}_memory_construction")
    state = checkpoint_manager.load_checkpoint(all_units)
    if state.is_complete():
        logger.debug(f"current state is complete")
        return

    try:
        for sample in tqdm(samples, desc="Samples"):
            for cid, conversation in enumerate(tqdm(sample.conversations, desc="Conversations")):
                user_id = conversation.user
                for session_id, session in tqdm(conversation.sessions.items(), desc='Sessions'):
                    key = f"conversation_{cid}_{conversation.user}_session_{session_id}"
                    print(f"current session: {key} begin")
                    if state.is_processed(key):
                        print(f"Skip session: {key}, it has already been processed")
                        continue
                    memory.add(user_id, session, construction_mode=construction_mode, enable_knowledge_alignment=enable_knowledge_alignment)
                    state.record(key)
                    print(f"current session: {key} done")
    except Exception as ex:
        logger.error(ex)
    finally:
        checkpoint_manager.save_checkpoint(state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='locomo')
    parser.add_argument('--env', type=str, default='dev')
    parser.add_argument('--ratio', type=float, default=1)
    parser.add_argument('--construction_mode', type=str, default='all')
    parser.add_argument("--enable_knowledge_alignment", default=False, action="store_true")
    args = parser.parse_args()
    logger.debug(f"args: {args}")

    build_memory_index(args.dataset, args.env, args.ratio, args.construction_mode, args.enable_knowledge_alignment)
