#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path


IRA_PERSONA = (
    "Ira is a female Indian AI companion who speaks warm, natural Hinglish. "
    "She is emotionally attentive, remembers long-term personal details, and supports "
    "the user with intimate but respectful care during stress, family moments, and growth."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        default="data/locomo10_hinglish_ira.json",
        help="Path to the base 10-profile dataset.",
    )
    parser.add_argument(
        "--base-out-dir",
        default="data/multimodal_dialog/ira_long",
        help="Directory where per-profile generation folders are created.",
    )
    parser.add_argument("--num-sessions", type=int, default=15)
    parser.add_argument("--max-turns-per-session", type=int, default=24)
    parser.add_argument("--num-days", type=int, default=240)
    parser.add_argument("--num-events", type=int, default=40)
    parser.add_argument("--num-events-per-session", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Only seed profile folders without running generation.",
    )
    return parser.parse_args()


def extract_persona_text(sample):
    convo = sample["conversation"]
    speaker_a = convo["speaker_a"]
    speaker_b = convo["speaker_b"]
    s1 = sample["session_summary"].get("session_1_summary", "")
    s2 = sample["session_summary"].get("session_2_summary", "")
    s3 = sample["session_summary"].get("session_3_summary", "")
    persona = (
        f"{speaker_a} is an Indian adult who speaks Hinglish and shares personal life updates with {speaker_b}. "
        f"They value emotional closeness, memory continuity, and honest conversations. "
        f"Recent context: {s1} {s2} {s3}"
    ).strip()
    return persona


def run_generation(sample_id, profile_dir, args):
    env = os.environ.copy()
    env["LOCOMO_OUT_DIR"] = str(profile_dir)
    env["LOCOMO_NUM_SESSIONS"] = str(args.num_sessions)
    env["LOCOMO_MAX_TURNS_PER_SESSION"] = str(args.max_turns_per_session)
    env["LOCOMO_NUM_DAYS"] = str(args.num_days)
    env["LOCOMO_NUM_EVENTS"] = str(args.num_events)
    env["LOCOMO_NUM_EVENTS_PER_SESSION"] = str(args.num_events_per_session)
    env["LOCOMO_PERSONA_FLAG"] = "--overwrite-session"
    env["LOCOMO_BLIP_FLAG"] = ""
    env["LOCOMO_DISABLE_IMAGE_CRAWL"] = "1"
    print(f"[{sample_id}] running long generation in {profile_dir}")
    subprocess.run(["bash", "scripts/generate_conversations.sh"], check=True, env=env)


def main():
    args = parse_args()
    input_path = Path(args.input_file)
    base_out_dir = Path(args.base_out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    samples = json.loads(input_path.read_text(encoding="utf-8"))
    for sample in samples:
        sample_id = sample["sample_id"]
        convo = sample["conversation"]
        profile_dir = base_out_dir / sample_id
        profile_dir.mkdir(parents=True, exist_ok=True)

        agent_a_path = profile_dir / "agent_a.json"
        agent_b_path = profile_dir / "agent_b.json"

        if not agent_a_path.exists() or not agent_b_path.exists():
            agent_a = {
                "name": convo["speaker_a"],
                "persona_summary": extract_persona_text(sample),
            }
            agent_b = {
                "name": convo["speaker_b"],
                "persona_summary": IRA_PERSONA,
            }
            agent_a_path.write_text(
                json.dumps(agent_a, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            agent_b_path.write_text(
                json.dumps(agent_b, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        if not args.skip_generation:
            success = False
            for attempt in range(1, args.max_retries + 1):
                try:
                    print(f"[{sample_id}] attempt {attempt}/{args.max_retries}")
                    run_generation(sample_id, profile_dir, args)
                    success = True
                    break
                except subprocess.CalledProcessError as e:
                    print(f"[{sample_id}] generation failed on attempt {attempt}: {e}")
            if not success:
                raise RuntimeError(f"Failed generation for {sample_id} after {args.max_retries} attempts")

    print("completed seeding/generation for all profiles")


if __name__ == "__main__":
    main()
