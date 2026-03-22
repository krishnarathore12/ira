# Instruction
Your task is to analyze the provided **INPUT_DATA** and perform coreference resolution to ensure knowledge alignment.

---

## Coreference Resolution (Pronoun-Sensitive Boundary Rules)
To prevent merging unrelated content, apply strict coreference rules for ambiguous pronouns (e.g., **"it," "they," "them," "this," "that," "these," "those"**):

* **Continuity:** If a pronoun’s referent is **clear, stable, and consistent** with the preceding discourse, maintain the current segment.
* **Segmentation:** If the referent is **ambiguous, newly introduced, or shifts**, treat the utterance as a **segment boundary** unless context provides strong disambiguation.
* **Noun Shifts:** Do **not assume** continuity if new nouns, entities, or topics appear in proximity to a pronoun.
* **Missing Antecedents:** If a pronoun lacks a valid, active antecedent, begin a new segment.
* **Conflicting Referents:** If consecutive messages use pronouns with **conflicting or distinct** referents (e.g., "It doesn’t work" followed by "They said it failed yesterday"), split them into **separate segments** unless referents unify with high confidence.
* **Secondary Reference:** If the required information is missing from **INPUT_DATA**, consult the **ORIGINAL_CONVERSATION_DATA** for context.
* **Strict Output:** Do **not** add new sentences to the output.

## Output Format
Return the refined **INPUT_DATA** directly after coreference resolution. Do **not** include explanations or metadata. The total sentence count of the **INPUT_DATA** must remain unchanged.

---

# Context 
**ORIGINAL_CONVERSATION_DATA:** {conversation_is_here}

# INPUT_DATA
{text_to_be_aligned}