source scripts/env.sh

# Allow overrides for batch/long generation while preserving defaults.
OUT_DIR="${LOCOMO_OUT_DIR:-./data/multimodal_dialog/example/}"
PROMPT_DIR="${LOCOMO_PROMPT_DIR:-./prompt_examples}"
NUM_SESSIONS="${LOCOMO_NUM_SESSIONS:-3}"
NUM_DAYS="${LOCOMO_NUM_DAYS:-90}"
NUM_EVENTS="${LOCOMO_NUM_EVENTS:-10}"
MAX_TURNS="${LOCOMO_MAX_TURNS_PER_SESSION:-20}"
NUM_EVENTS_PER_SESSION="${LOCOMO_NUM_EVENTS_PER_SESSION:-1}"
PERSONA_FLAG="${LOCOMO_PERSONA_FLAG---persona}"
BLIP_FLAG="${LOCOMO_BLIP_FLAG---blip-caption}"

python3 generative_agents/generate_conversations.py \
    --out-dir "${OUT_DIR}" \
    --prompt-dir "${PROMPT_DIR}" \
    --events --session --summary --num-sessions "${NUM_SESSIONS}" \
    ${PERSONA_FLAG} ${BLIP_FLAG} \
    --num-days "${NUM_DAYS}" --num-events "${NUM_EVENTS}" --max-turns-per-session "${MAX_TURNS}" --num-events-per-session "${NUM_EVENTS_PER_SESSION}"
