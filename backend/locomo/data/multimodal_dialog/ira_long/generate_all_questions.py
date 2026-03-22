import os
import json
import urllib.request
import urllib.error
import re
import time

API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

BASE_DIR = "/home/rathore/Desktop/github-repo/ira/backend/locomo/data/multimodal_dialog/ira_long"
DIRECTORIES = [
    "ira-conv-05",
    "ira-conv-06",
    "ira-conv-07",
    "ira-conv-08",
    "ira-conv-09",
    "ira-conv-10"
]

def load_example():
    with open(os.path.join(BASE_DIR, "ira-conv-01", "questions.json"), "r") as f:
        return f.read()

def generate_questions_for_dir(dir_name):
    print(f"Processing {dir_name}...")
    agent_a_path = os.path.join(BASE_DIR, dir_name, "agent_a.json")
    
    with open(agent_a_path, "r", encoding="utf-8") as f:
        agent_data = json.load(f)
        
    persona_name = agent_data.get("name", "Unknown")
    
    # We will just pass the entire agent_data as JSON string to the LLM
    # since it contains the persona summary, graph, and all session dialogues and facts.
    agent_json_str = json.dumps(agent_data, ensure_ascii=False)
    
    example_json = load_example()
    
    prompt = f"""
You are an expert evaluator of long-term memory in AI agents.
I am going to provide you with the full conversation data, event graph, and facts for a user persona named '{persona_name}'.
Your task is to create a list of EXACTLY 30 memory-testing questions (3 per category across 10 categories) formatted as JSON, just like the provided example.

Categories:
Critical tier:
- C1 (explicit recall): baseline retrieval of direct facts
- C2 (correction/supersession): testing superseding stale memory or inline corrections
- C4 (uncertainty/honesty): testing things NOT mentioned in the text (the answer should be "Not mentioned...")
- C9 (multi-user isolation): testing zero-tolerance for mixing up the user's data with typical defaults

High tier:
- C3 (temporal ordering): cause-and-effect timelines, before/after
- C5 (multi-turn continuity): facts that survive and connect across multiple explicit sessions
- C6 (sensitive memory): delicate emotional states, family loss, deep memories
- C7 (entity nuance): exact details regarding names, specific items, specific relationships

Medium tier:
- C8 (epistemic honesty): separate exactly stated facts from assumed inferences (e.g. "Can we infer he plays every weekend?" -> "No, only stated...")
- C10 (tone/warmth): "How should Ira respond when..." or "What is the most contextually accurate and warm way..."

Here is an example layout and style of the JSON for another persona 'Aarav':
```json
{example_json}
```

Now, based on the following JSON data for persona '{persona_name}', generate the new 30 questions.
Only output the raw valid JSON. Do not include markdown codeblocks (no ```json).

Context for {persona_name}:
{agent_json_str}
"""
    
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You output only valid JSON. No markdown formatting."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(data).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
    )
    
    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read().decode("utf-8"))
        ans = result["choices"][0]["message"]["content"]
        
        ans = ans.strip()
        if ans.startswith("```json"):
            ans = ans[7:]
        if ans.startswith("```"):
            ans = ans[3:]
        if ans.endswith("```"):
            ans = ans[:-3]
            
        out_path = os.path.join(BASE_DIR, dir_name, "questions.json")
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(ans)
        print(f"Successfully generated {out_path}")
        
    except urllib.error.HTTPError as e:
        print(f"HTTP Error for {dir_name}: {e.code} - {e.read().decode('utf-8')}")
    except Exception as e:
        print(f"Error for {dir_name}: {str(e)}")

for d in DIRECTORIES:
    generate_questions_for_dir(d)
    time.sleep(2) # be nice to rate limits
