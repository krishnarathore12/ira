import json

from loguru import logger

from himem.memory.utils import remove_code_blocks
from himem.utils.factory import LlmFactory


class KnowledgeConflictDetector:
    def __init__(self, config):
        current_config = config.components['knowledge_conflict_detector'].config
        self.enabled_llm_provider = current_config["llm_provider"]
        self.llm_config = config.llm_providers[self.enabled_llm_provider].config
        self.llm = LlmFactory.create(self.enabled_llm_provider, self.llm_config)
        self.config = current_config
        knowledge_conflict_detection_prompt_path = self.config.get('knowledge_conflict_detection_prompt_path')
        with open(knowledge_conflict_detection_prompt_path, "r", encoding="utf-8") as f:
            self.knowledge_conflict_detection_prompt = f.read()

    def resolve(self, new_extracted_user_related_notes, retrieved_previous_memories):
        logger.info(f"Total existing memories: {len(retrieved_previous_memories)}")
        # mapping UUIDs with integers for handling UUID hallucinations
        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_previous_memories):
            temp_uuid_mapping[item['content']] = item["id"]
            retrieved_previous_memories[idx]["id"] = str(idx)

        new_memories_with_actions = self.__predict_action(new_extracted_user_related_notes, retrieved_previous_memories)
        return new_memories_with_actions, temp_uuid_mapping

    def __predict_action(self, new_extracted_notes, previous_memories):
        new_memories_with_actions = {}
        if not new_extracted_notes:
            return new_memories_with_actions
        new_memory_parts = "\n".join([json.dumps(note) for note in new_extracted_notes])
        function_calling_prompt = self.get_update_memory_messages(
            previous_memories, new_memory_parts,
        )

        try:
            response, _ = self.llm.generate_response(
                messages=[{"role": "user", "content": function_calling_prompt}],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.error(f"Error in new memory actions response: {e}")
            response = ""

        try:
            if not response or not response.strip():
                logger.warning("Empty response from LLM, no memories to extract")
                new_memories_with_actions = {}
            else:
                response = remove_code_blocks(response)
                new_memories_with_actions = json.loads(response)
        except Exception as e:
            logger.error(f"Invalid JSON response: {e}, response:{response}")
            new_memories_with_actions = {}

        return new_memories_with_actions

    def get_update_memory_messages(self, previous_memories, new_memory_parts):
        if previous_memories:
            current_memory_part = f"""
        Below is the current content of my memory which I have collected till now. You have to update it in the following format only:

        ```
        {previous_memories}
        ```

        """
        else:
            current_memory_part = """
        Current memory is empty.

        """

        return f"""{self.knowledge_conflict_detection_prompt}

        {current_memory_part}

        The new retrieved facts are mentioned in the triple backticks. You have to analyze the new retrieved facts and determine whether these facts should be added, updated, or deleted in the memory.

        ```
        {new_memory_parts}
        ```

        You must return your response in the following JSON structure only:

        {{
            "memory" : [
                {{
                    "id" : "<ID of the memory>",                # Use existing ID for updates/deletes, or new ID for additions
                    "text" : "<Content of the memory>",         # Content of the memory
                    "category" : "<Category of the memory>",    # Content of the memory
                    "event" : "<Operation to be performed>",    # Must be "ADD", "UPDATE", "DELETE", or "NONE"
                    "old_memory" : "<Old memory content>"       # Required only if the event is "UPDATE"
                }},
                ...
            ]
        }}

        Follow the instruction mentioned below:
        - Do not return anything from the custom few shot prompts provided above.
        - If the current memory is empty, then you have to add the new retrieved facts to the memory.
        - You should return the updated memory in only JSON format as shown below. The memory key should be the same if no changes are made.
        - If there is an addition, generate a new key and add the new memory corresponding to it.
        - If there is a deletion, the memory key-value pair should be removed from the memory.
        - If there is an update, the ID key should remain the same and only the value needs to be updated.

        Do not return anything except the JSON format.
        """
