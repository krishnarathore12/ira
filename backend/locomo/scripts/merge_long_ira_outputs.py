#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-base-dir",
        default="data/multimodal_dialog/ira_long",
        help="Base directory containing per-profile generation folders.",
    )
    parser.add_argument(
        "--output-file",
        default="data/locomo10_hinglish_ira_long.json",
        help="Merged LoCoMo-style output path.",
    )
    return parser.parse_args()


def session_numbers(agent):
    nums = []
    for k in agent.keys():
        m = re.fullmatch(r"session_(\d+)", k)
        if m:
            nums.append(int(m.group(1)))
    return sorted(nums)


def to_event_summary(agent_a, agent_b, sess_nums):
    out = {}
    for n in sess_nums:
        key = f"events_session_{n}"
        events_a = agent_a.get(key, [])
        events_b = agent_b.get(key, [])
        day = ""
        if agent_a.get(f"session_{n}_date_time"):
            dt = agent_a[f"session_{n}_date_time"]
            if " on " in dt:
                day = dt.split(" on ", 1)[1]
        out[key] = {
            agent_a["name"]: [e.get("sub-event", e.get("sub_event", "")) for e in events_a if isinstance(e, dict)],
            agent_b["name"]: [e.get("sub-event", e.get("sub_event", "")) for e in events_b if isinstance(e, dict)],
            "date": day,
        }
    return out


def to_observation(agent, sess_nums):
    out = {}
    for n in sess_nums:
        fkey = f"session_{n}_facts"
        okey = f"session_{n}_observation"
        facts = agent.get(fkey, {})
        obs = {}
        for speaker, entries in facts.items():
            cleaned = []
            if isinstance(entries, list):
                for item in entries:
                    if isinstance(item, list) and len(item) == 2:
                        cleaned.append([item[0], item[1]])
                    elif isinstance(item, tuple) and len(item) == 2:
                        cleaned.append([item[0], item[1]])
            obs[speaker] = cleaned
        out[okey] = obs
    return out


def to_session_summary(agent, sess_nums):
    out = {}
    for n in sess_nums:
        out[f"session_{n}_summary"] = agent.get(f"session_{n}_summary", "")
    return out


def to_conversation(agent_a, agent_b, sess_nums):
    conv = {
        "speaker_a": agent_a["name"],
        "speaker_b": agent_b["name"],
    }
    for n in sess_nums:
        conv[f"session_{n}_date_time"] = agent_a.get(f"session_{n}_date_time", "")
        conv[f"session_{n}"] = agent_a.get(f"session_{n}", [])
    return conv


def main():
    args = parse_args()
    base = Path(args.input_base_dir)
    merged = []

    for profile_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        agent_a_path = profile_dir / "agent_a.json"
        agent_b_path = profile_dir / "agent_b.json"
        if not agent_a_path.exists() or not agent_b_path.exists():
            continue

        agent_a = json.loads(agent_a_path.read_text(encoding="utf-8"))
        agent_b = json.loads(agent_b_path.read_text(encoding="utf-8"))
        sess_nums = session_numbers(agent_a)
        if not sess_nums:
            continue

        sample = {
            "qa": [],
            "conversation": to_conversation(agent_a, agent_b, sess_nums),
            "event_summary": to_event_summary(agent_a, agent_b, sess_nums),
            "observation": to_observation(agent_a, sess_nums),
            "session_summary": to_session_summary(agent_a, sess_nums),
            "sample_id": profile_dir.name,
        }
        merged.append(sample)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(merged)} samples to {out_path}")


if __name__ == "__main__":
    main()
