import json
from pathlib import Path
from typing import Union, List

import pendulum

from himem.dataset.model import Sample, QA, Conversation, Turn, Session
from himem.utils.base import DEFAULT_DATE_FORMAT


def load_longmemeval_dataset(file_path: Union[str, Path], ratio: float = 1) -> List[Sample]:
    """
    Load the LongMemEval dataset from a JSON file, including image-based content by using captions.
    Args:
        file_path: Path to the JSON file containing the dataset
    Returns:
        List of LoCoMoSample objects containing the parsed data
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    total_qa = 0
    qa_counts_per_sample = []
    total_nums = len(data)
    data = data[:int(ratio * total_nums)]
    for sample_idx, sample in enumerate(data):
        try:
            question_id = sample['question_id']
            question = sample['question']
            answer = sample['answer']
            category = sample['question_type']
            evidences = sample['answer_session_ids']
            qa = QA(question, answer, evidences, category)
            haystack_dates = sample['haystack_dates']
            haystack_session_ids = sample['haystack_session_ids']
            haystack_sessions = sample['haystack_sessions']
            user = "Xiaowu"
            sessions = {}
            for session_idx, session in enumerate(haystack_sessions):
                session_date = haystack_dates[session_idx]
                session_date = pendulum.from_format(session_date, "YYYY/MM/DD (ddd) H:mm").format(
                    DEFAULT_DATE_FORMAT)
                session_id = haystack_session_ids[session_idx]
                sessions[session_id] = create_session_from_original_data(session_id, session_date, session)
            conversation = Conversation(user=user, sessions=sessions)

            samples.append(Sample(sample_id=question_id, qa=[qa], conversations=[conversation]))

            sample_qa_count = 1
            total_qa += sample_qa_count
            qa_counts_per_sample.append(sample_qa_count)

            # Print statistics for this sample
            print(f"\nSample {sample_idx}:")
            print(f"  Total QAs: {sample_qa_count}")
            print(f"  Total Sessions: {len(sessions)}")
        except Exception as ex:
            print(f"\nSample {sample_idx}: error: {ex}")
    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"Total QAs: {total_qa}")
    print(f"Average QAs per sample: {total_qa / len(samples):.2f}")
    print(f"Min QAs in a sample: {min(qa_counts_per_sample)}")
    print(f"Max QAs in a sample: {max(qa_counts_per_sample)}")
    return samples


def create_session_from_original_data(session_idx, session_date, session):
    turns = []
    for turn_idx, turn in enumerate(session):
        turns.append(Turn(f"{session_idx}_{turn_idx}", turn["role"], turn["content"]))
    return Session(session_idx, session_date, turns)
