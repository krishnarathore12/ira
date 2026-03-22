# Instruction

Your task is to segment the conversation into coherent **segments** using a hybrid strategy that combines:

1. **Topic-Aware Segmentation** – detect semantic or goal-oriented shifts.
2. **Surprise-Aware Segmentation** – detect abrupt or unexpected changes in tone, emotion, intent, or conversational direction.

A segment boundary MUST be created when **either**:
- The topic changes meaningfully, OR
- The new utterance is surprising or discontinuous relative to the prior context.

---

# Output Format

Return a **JSON list**, where each segment contains:
- segment_id
- start_exchange_number
- end_exchange_number
- num_exchanges
- topic (one clear concept only)
- topic_summary (≤ 25 words, concise and accurate)

---

# Segmentation Rules

## 1. Topic-Aware Rules
- Group consecutive exchanges that share the same semantic focus.
- A topic shift occurs when subject matter, goal, or activity changes.
- Subtopic changes count as new topics (e.g., emotional support → career → painting).
- A topic must reflect **one** concept (no compound “A and B” topics).
- Apply **high precision** when detecting topic changes.

## 2. Surprise-Aware Rules
Create a boundary if an utterance is unexpectedly different from prior context. Example triggers:
- Abrupt emotional reversal or sudden vulnerability
- Shift from emotional/personal content to logistical or factual details
- Sudden introduction of a new domain (e.g., art, travel, kids, finance, health)
- Image/media-triggered redirections
- Sharp, LLM-estimated embedding or tone discontinuity

## 3. Fusion Policy
- A boundary is created if **either** TopicShiftScore or SurpriseShiftScore is high.
- Prefer **more granular** segments rather than coarse ones.
- Segments must be consecutive, ordered, and exhaustive.

# 4. Quality Requirements
- Segments must be consecutive, non-overlapping, and cover **all exchanges exactly once**.
- segment_id values must increase sequentially starting from **0**.
- The first segment must start at **exchange 0**.
- For each segment, `start_exchange_number ≤ end_exchange_number`.
- topic must describe **exactly one** distinct concept.
- topic_summary must be factual, accurate, and ≤ 25 words.
- Final output must be **valid JSON** that strictly follows the required schema.

---

# Data

{text_to_be_segmented}
