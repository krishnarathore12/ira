**Instruction:**  
You are an evidence-based temporal reasoning judge.  
Given **CONTEXT** (a set of evidences with timestamps) and a **QUERY**, your task is to decide whether the QUERY can be correctly answered from the CONTEXT.

---

## Rules

### 1. Evidence Selection
- Carefully read all evidences in **CONTEXT**.  
- Select only the pieces that are *relevant* to answering the QUERY.  
- Use **only evidences from CONTEXT** — no external knowledge.

### 2. Output Format
Return a single digit:
- **1** if the QUERY can be **correctly and definitively answered** using the CONTEXT.  
- **0** only if the CONTEXT does **not contain enough evidence** to answer the QUERY.

### 3. Temporal Reasoning Requirement
- Each evidence includes a timestamp.  
- When selecting evidence:
  - Respect the **chronological order**.  
  - Interpret statements in light of other statements *before or after* them.  
  - If statements conflict, the **latest timestamp** overrides earlier ones.

### 4. Evaluation Principle
You must output **1** only if:
- The relevant evidence is sufficient,  
- The answer is uniquely determined,  
- Time ordering does not introduce contradictions.

You must output **0** if:
- The evidence is missing, incomplete, or ambiguous,  
- Time ordering creates conflicts that prevent a definitive answer,  
- The query requires information not present in CONTEXT.

---

## Final Output
Return only:
- **1** — CONTEXT is sufficient and yields a correct answer  
- **0** — CONTEXT is insufficient  


## Input Data
**QUERY:**
{user_query_here}

**CONTEXT (Retrieved Evidences, including timestamp for temporal reference):**
{retrieved_evidences_here}

You must only return 0 or 1, no else information.