import json

from loguru import logger

from himem.dataset.model import Session
from himem.memory.utils import parse_messages, extract_json, parse_notes_from_response
from himem.utils.factory import LlmFactory


class Extractor:
    def __init__(self, config):
        self.config = config.components['extractor'].config
        self.enabled_llm_provider = self.config.get('llm_provider')
        self.llm_config = config.llm_providers[self.enabled_llm_provider].config
        self.llm = LlmFactory.create(self.enabled_llm_provider, self.llm_config)

        stage_1_prompt_path = self.config.get('stage_1_prompt_path')
        with open(stage_1_prompt_path, "r", encoding="utf-8") as f:
            self.stage_1_prompt = f.read()
        stage_2_prompt_path = self.config.get('stage_2_prompt_path')
        with open(stage_2_prompt_path, "r", encoding="utf-8") as f:
            self.stage_2_prompt = f.read()
        stage_3_prompt_path = self.config.get('stage_3_prompt_path')
        with open(stage_3_prompt_path, "r", encoding="utf-8") as f:
            self.stage_3_prompt = f.read()
        knowledge_extraction_prompt_path = self.config.get('knowledge_extraction_prompt_path')
        with open(knowledge_extraction_prompt_path, "r", encoding="utf-8") as f:
            self.knowledge_extraction_prompt = f.read()

        self.note_extraction_response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "fact_extraction_output",
                "schema": {
                    "type": "object",
                    "properties": {
                        "notes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "timestamp": {"type": "string"},
                                    "category": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                                "required": [
                                    "timestamp",
                                    "category",
                                    "content",
                                ],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["notes"],
                    "additionalProperties": False
                }
            }
        }

        self.knowledge_extraction_response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "fact_extraction_output",
                "schema": {
                    "type": "object",
                    "properties": {
                        "notes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "timestamp": {"type": "string"},
                                    "category": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                                "required": [
                                    "timestamp",
                                    "category",
                                    "content",
                                ],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["notes"],
                    "additionalProperties": False
                }
            }
        }

    def stage_1_extraction(self, user_id: str, session: Session):
        messages = []
        for turn in session.turns:
            if turn.role == "user":
                messages.append({'role': turn.role, 'content': turn.content})
        parsed_messages = parse_messages(messages)

        system_prompt = self.stage_1_prompt.format(
            REFERENCE_TIME_IS_HERE=session.date_time,
            USER_IS_HERE=user_id,
        )
        user_prompt = f"Conversation:\n{parsed_messages}"
        response, _ = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=self.note_extraction_response_format
        )
        return parse_notes_from_response(response)

    def stage_2_extraction(self, user_id: str, session: Session):
        messages = []
        for turn in session.turns:
            if turn.role == "user":
                messages.append({'role': turn.role, 'content': turn.content})
        parsed_messages = parse_messages(messages)

        system_prompt = self.stage_2_prompt.format(
            REFERENCE_TIME_IS_HERE=session.date_time,
            USER_IS_HERE=user_id,
        )
        user_prompt = f"Conversation:\n{parsed_messages}"
        response, _ = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=self.note_extraction_response_format
        )
        return parse_notes_from_response(response)

    def stage_3_refine(self, user_id: str, session: Session, extracted_notes: list):
        logger.debug(f"Enabling knowledge alignment in note memory for {user_id} over {session.session_id}")
        messages = []
        for turn in session.turns:
            if turn.role == "user":
                messages.append({'role': turn.role, 'content': turn.content})
        parsed_messages = parse_messages(messages)

        system_prompt = self.stage_3_prompt.format(
            REFERENCE_TIME_IS_HERE=session.date_time,
            USER_IS_HERE=user_id,
            CONVERSATION_IS_HERE=parsed_messages,
        )
        user_prompt = f"Extracted Facts:\n{extracted_notes}"
        response, _ = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=self.note_extraction_response_format
        )
        return parse_notes_from_response(response)

    def extract_notes(self, user_id: str, session: Session, enable_knowledge_alignment=False):
        facts = self.stage_1_extraction(user_id, session)
        user_infos = self.stage_2_extraction(user_id, session)
        extracted_notes = []
        extracted_notes.extend(facts)
        extracted_notes.extend(user_infos)
        if enable_knowledge_alignment:
            extracted_notes = self.stage_3_refine(user_id, session, extracted_notes)
        return extracted_notes

    def knowledge_extraction(self, question: str, retrieved_context: list):
        user_prompt = self.knowledge_extraction_prompt.format(
            question_is_here=question,
            input_data_is_here=retrieved_context,
        )
        response, _ = self.llm.generate_response(
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            response_format=self.knowledge_extraction_response_format
        )
        result = json.loads(response)
        return result['notes']
