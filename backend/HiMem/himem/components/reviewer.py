import uuid
from copy import deepcopy

from himem.utils.factory import LlmFactory, EmbedderFactory, VectorStoreFactory


class Reviewer:
    def __init__(self, config):
        self.config = config.components['reviewer'].config
        self.enabled_llm_provider = self.config.get('llm_provider')
        self.llm_config = config.llm_providers[self.enabled_llm_provider].config
        self.llm = LlmFactory.create(self.enabled_llm_provider, self.llm_config)

        evaluation_prompt_path = self.config.get('evaluation_prompt_path')
        with open(evaluation_prompt_path, "r", encoding="utf-8") as f:
            self.evaluation_prompt = f.read()

    def evaluate(self, query, retrieved_evidences):
        prompt = self.evaluation_prompt.format(
            user_query_here=query,
            retrieved_evidences_here=retrieved_evidences
        )
        response, _ = self.llm.generate_response([{'role': 'user', 'content': prompt}])
        if not response:
            response = 0
        return int(response)
