import json
import os

start_conv = 5
end_conv = 10
base_dir = '/home/rathore/Desktop/github-repo/ira/backend/locomo/data/multimodal_dialog/ira_long'
output_file = os.path.join(base_dir, 'facts.txt')

with open(output_file, 'w') as out:
    for i in range(start_conv, end_conv + 1):
        conv_dir = f'ira-conv-{i:02d}'
        out.write(f"=== {conv_dir} ===\n")
        
        # We will check agent_a.json and agent_b.json
        for agent_file in ['agent_a.json', 'agent_b.json']:
            file_path = os.path.join(base_dir, conv_dir, agent_file)
            if not os.path.exists(file_path):
                continue
                
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    name = data.get('name', 'Unknown')
                    
                    # Search for facts in all session_X_facts
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if key.endswith('_facts') and isinstance(value, dict):
                                out.write(f"  [{agent_file}] Agent Name: {name} - {key}:\n")
                                for person, facts in value.items():
                                    out.write(f"      {person} Facts:\n")
                                    for fact in facts:
                                        if isinstance(fact, list) and len(fact) > 0:
                                            out.write(f"        - {fact[0]}\n")
            except Exception as e:
                out.write(f"Error reading {file_path}: {e}\n")
        out.write("\n")
