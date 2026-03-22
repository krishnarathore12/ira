import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import pendulum

from himem.dataset.model import QA, Conversation, Turn, Session, Sample
from himem.utils.base import DEFAULT_DATE_FORMAT

category_mapping = {4: 'single-hop', 1: 'multi-hop', 2: 'temporal-reasoning', 3: 'open-domain',
                    5: 'adversarial'}


def load_locomo_dataset(file_path: Union[str, Path], ratio: float = 1) -> List[Sample]:
    """
    Load the LoComo dataset from a JSON file, including image-based content by using captions.
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
    total_image_qa = 0
    qa_counts_per_sample = []
    total_nums = len(data)
    data = data[:int(ratio * total_nums)]
    for sample_idx, sample in enumerate(data):
        try:
            # Parse QA data
            qa_list = []
            sample_qa_count = 0
            sample_image_qa_count = 0

            for qa_idx, qa in enumerate(sample["qa"]):
                try:
                    category = int(qa.get("category"))
                    if category == 5:
                        continue
                    # Check if QA has image evidence
                    has_image_evidence = False
                    for evidence_id in qa.get("evidence", []):
                        if ":" not in evidence_id:
                            continue
                        turn_id = evidence_id.split(":")[1]
                        for session in sample["conversation"].values():
                            if isinstance(session, list):
                                for turn in session:
                                    if turn.get("dia_id", "").endswith(turn_id):
                                        if "img_url" in turn or "blip_caption" in turn:
                                            has_image_evidence = True
                                            break

                    if has_image_evidence:
                        sample_image_qa_count += 1

                    adversarial_answer = qa.get("adversarial_answer", None)
                    if adversarial_answer:
                        answer = adversarial_answer
                    else:
                        answer = qa.get("answer"),
                    qa_obj = QA(
                        question=qa["question"],
                        answer=answer,
                        evidences=qa.get("evidence", []),
                        category=category_mapping.get(qa.get("category")),
                    )
                    qa_list.append(qa_obj)
                    sample_qa_count += 1

                except KeyError as e:
                    print(f"Error in sample {sample_idx}, QA pair {qa_idx}:")
                    print(f"QA data: {qa}")
                    raise e
                except Exception as e:
                    print(f"Unexpected error in sample {sample_idx}, QA pair {qa_idx}:")
                    print(f"QA data: {qa}")
                    raise e

            # Parse conversation
            conv_data = sample["conversation"]
            speaker_a = conv_data["speaker_a"]
            speaker_b = conv_data["speaker_b"]
            conversations = []
            for user in [speaker_a, speaker_b]:
                conversations.append(parse_conversation(str(sample_idx), user, conv_data))

            sample_obj = Sample(
                sample_id=str(sample_idx),
                qa=qa_list,
                conversations=conversations,
            )
            samples.append(sample_obj)

            total_qa += sample_qa_count
            total_image_qa += sample_image_qa_count
            qa_counts_per_sample.append(sample_qa_count)

            # Print statistics for this sample
            print(f"\nSample {sample_idx}:")
            print(f"  Total QAs: {sample_qa_count}")
            print(f"  QAs with image evidence: {sample_image_qa_count}")

        except Exception as e:
            print(f"Error processing sample {sample_idx}:")
            print(str(e))
            raise e

    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"Total QAs: {total_qa}")
    print(f"Total QAs with image evidence: {total_image_qa}")
    print(f"Average QAs per sample: {total_qa / len(samples):.2f}")
    print(f"Min QAs in a sample: {min(qa_counts_per_sample)}")
    print(f"Max QAs in a sample: {max(qa_counts_per_sample)}")

    return samples


def parse_session(user: str, session_data: List[dict], session_id: str, date_time: str) -> Session:
    """Parse a single session's data, including turns with images by using their captions."""
    turns = []
    for turn in session_data:
        # For turns with images, combine caption and text
        text = turn.get("text", "")
        if "img_url" in turn and "blip_caption" in turn:
            if 'query' in turn:
                caption_text = f"This is {turn['blip_caption']}[{turn['query']}]."
            else:
                caption_text = f"This is {turn['blip_caption']}."
            if text:
                text = f"{caption_text} {text}"
            else:
                text = caption_text

        speaker = turn["speaker"]
        role = "assistant"
        if speaker == user:
            role = "user"
        turns.append(Turn(
            dia_id=turn["dia_id"],
            role=role,
            content=text
        ))
    return Session(session_id=session_id, date_time=date_time, turns=turns)


def parse_conversation(sample_idx: str, user: str, conv_data: dict) -> Conversation:
    """Parse conversation data."""
    sessions = {}
    for key, value in conv_data.items():
        if key.startswith("session_") and isinstance(value, list):
            session_id = f'{sample_idx}_{int(key.split("_")[1])}'
            original_date_time = conv_data.get(f"{key}_date_time")
            if original_date_time:
                try:
                    date_time = pendulum.from_format(original_date_time, "h:mm a on D MMMM, YYYY").format(
                        DEFAULT_DATE_FORMAT)
                    session = parse_session(user, value, session_id, date_time)
                    # Only add sessions that have turns after filtering
                    if session.turns:
                        sessions[session_id] = session
                except Exception as e:
                    print(f"Error parsing date_time {original_date_time}, {e}")
                    raise e

    return Conversation(
        user=user,
        sessions=sessions
    )


def get_dataset_statistics(samples: List[Sample]) -> Dict:
    """
    Get basic statistics about the text-only dataset.
    Args:
        samples: List of LoCoMoSample objects
    Returns:
        Dictionary containing various statistics about the dataset
    """
    stats = {
        "num_samples": len(samples),
        "total_qa_pairs": sum(len(sample.qa) for sample in samples),
        "total_sessions": sum(len(sample.conversation.sessions) for sample in samples),
        "total_turns": sum(
            sum(len(session.turns) for session in sample.conversation.sessions.values())
            for sample in samples
        ),
        "qa_with_adversarial": sum(
            sum(1 for qa in sample.qa if qa.category == "adversarial") for sample in samples
        )
    }
    return stats
