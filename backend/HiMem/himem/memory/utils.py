# Framework structure adapted from Mem0 (Apache 2.0 License)
import json
import re
from copy import deepcopy
from typing import Optional, Any, Dict

from himem.exceptions import ValidationError


def parse_messages(messages):
    response = ""
    for msg in messages:
        if msg["role"] == "system":
            response += f"system: {msg['content']}\n"
        if msg["role"] == "user":
            response += f"user: {msg['content']}\n"
        if msg["role"] == "assistant":
            response += f"assistant: {msg['content']}\n"
    return response


def remove_code_blocks(content: str) -> str:
    """
    Removes enclosing code block markers ```[language] and ``` from a given string.

    Remarks:
    - The function uses a regex pattern to match code blocks that may start with ``` followed by an optional language tag (letters or numbers) and end with ```.
    - If a code block is detected, it returns only the inner content, stripping out the markers.
    - If no code block markers are found, the original content is returned as-is.
    """
    pattern = r"^```[a-zA-Z0-9]*\n([\s\S]*?)\n```$"
    match = re.match(pattern, content.strip())
    match_res = match.group(1).strip() if match else content.strip()
    return re.sub(r"<think>.*?</think>", "", match_res, flags=re.DOTALL).strip()


def extract_json(text):
    """
    Extracts JSON content from a string, removing enclosing triple backticks and optional 'json' tag if present.
    If no code block is found, returns the text as-is.
    """
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = text  # assume it's raw JSON
    return json_str


def _build_filters_and_metadata(
        *,  # Enforce keyword-only arguments
        user_id: Optional[str] = None,
        actor_id: Optional[str] = None,  # For query-time filtering
        input_metadata: Optional[Dict[str, Any]] = None,
        input_filters: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Constructs metadata for storage and filters for querying based on session and actor identifiers.

    This helper supports multiple session identifiers (`user_id`, `agent_id`, and/or `run_id`)
    for flexible session scoping and optionally narrows queries to a specific `actor_id`. It returns two dicts:

    1. `base_metadata_template`: Used as a template for metadata when storing new memories.
       It includes all provided session identifier(s) and any `input_metadata`.
    2. `effective_query_filters`: Used for querying existing memories. It includes all
       provided session identifier(s), any `input_filters`, and a resolved actor
       identifier for targeted filtering if specified by any actor-related inputs.

    Actor filtering precedence: explicit `actor_id` arg → `filters["actor_id"]`
    This resolved actor ID is used for querying but is not added to `base_metadata_template`,
    as the actor for storage is typically derived from message content at a later stage.

    Args:
        user_id (Optional[str]): User identifier, for session scoping.
        actor_id (Optional[str]): Explicit actor identifier, used as a potential source for
            actor-specific filtering. See actor resolution precedence in the main description.
        input_metadata (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session identifiers for the storage metadata template. Defaults to an empty dict.
        input_filters (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session and actor identifiers for query filters. Defaults to an empty dict.

    Returns:
        tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - base_metadata_template (Dict[str, Any]): Metadata template for storing memories,
              scoped to the provided session(s).
            - effective_query_filters (Dict[str, Any]): Filters for querying memories,
              scoped to the provided session(s) and potentially a resolved actor.
    """

    base_metadata_template = deepcopy(input_metadata) if input_metadata else {}
    effective_query_filters = deepcopy(input_filters) if input_filters else {}

    # ---------- add all provided session ids ----------
    session_ids_provided = []

    if user_id:
        base_metadata_template["user_id"] = user_id
        effective_query_filters["user_id"] = user_id
        session_ids_provided.append("user_id")

    if not session_ids_provided:
        raise ValidationError(
            message="At least one of 'user_id', 'agent_id', or 'run_id' must be provided.",
            error_code="VALIDATION_001",
            details={"provided_ids": {"user_id": user_id, }},
            suggestion="Please provide at least one identifier to scope the memory operation."
        )

    # ---------- optional actor filter ----------
    resolved_actor_id = actor_id or effective_query_filters.get("actor_id")
    if resolved_actor_id:
        effective_query_filters["actor_id"] = resolved_actor_id

    return base_metadata_template, effective_query_filters


def _has_advanced_operators(filters: Dict[str, Any]) -> bool:
    """
    Check if filters contain advanced operators that need special processing.

    Args:
        filters: Dictionary of filters to check

    Returns:
        bool: True if advanced operators are detected
    """
    if not isinstance(filters, dict):
        return False

    for key, value in filters.items():
        # Check for platform-style logical operators
        if key in ["AND", "OR", "NOT"]:
            return True
        # Check for comparison operators (without $ prefix for universal compatibility)
        if isinstance(value, dict):
            for op in value.keys():
                if op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]:
                    return True
        # Check for wildcard values
        if value == "*":
            return True
    return False


def _process_metadata_filters(metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process enhanced metadata filters and convert them to vector store compatible format.

    Args:
        metadata_filters: Enhanced metadata filters with operators

    Returns:
        Dict of processed filters compatible with vector store
    """
    processed_filters = {}

    def process_condition(key: str, condition: Any) -> Dict[str, Any]:
        if not isinstance(condition, dict):
            # Simple equality: {"key": "value"}
            if condition == "*":
                # Wildcard: match everything for this field (implementation depends on vector store)
                return {key: "*"}
            return {key: condition}

        result = {}
        for operator, value in condition.items():
            # Map platform operators to universal format that can be translated by each vector store
            operator_map = {
                "eq": "eq", "ne": "ne", "gt": "gt", "gte": "gte",
                "lt": "lt", "lte": "lte", "in": "in", "nin": "nin",
                "contains": "contains", "icontains": "icontains"
            }

            if operator in operator_map:
                result[key] = {operator_map[operator]: value}
            else:
                raise ValueError(f"Unsupported metadata filter operator: {operator}")
        return result

    for key, value in metadata_filters.items():
        if key == "AND":
            # Logical AND: combine multiple conditions
            if not isinstance(value, list):
                raise ValueError("AND operator requires a list of conditions")
            for condition in value:
                for sub_key, sub_value in condition.items():
                    processed_filters.update(process_condition(sub_key, sub_value))
        elif key == "OR":
            # Logical OR: Pass through to vector store for implementation-specific handling
            if not isinstance(value, list) or not value:
                raise ValueError("OR operator requires a non-empty list of conditions")
            # Store OR conditions in a way that vector stores can interpret
            processed_filters["$or"] = []
            for condition in value:
                or_condition = {}
                for sub_key, sub_value in condition.items():
                    or_condition.update(process_condition(sub_key, sub_value))
                processed_filters["$or"].append(or_condition)
        elif key == "NOT":
            # Logical NOT: Pass through to vector store for implementation-specific handling
            if not isinstance(value, list) or not value:
                raise ValueError("NOT operator requires a non-empty list of conditions")
            processed_filters["$not"] = []
            for condition in value:
                not_condition = {}
                for sub_key, sub_value in condition.items():
                    not_condition.update(process_condition(sub_key, sub_value))
                processed_filters["$not"].append(not_condition)
        else:
            processed_filters.update(process_condition(key, value))

    return processed_filters


def parse_notes_from_response(response):
    try:
        result = json.loads(response)
    except json.decoder.JSONDecodeError:
        extracted_json = extract_json(response)
        new_retrieved_facts = json.loads(extracted_json)["notes"]
        return new_retrieved_facts
    except Exception as e:
        print(f"Exception while parsing response: {e}")
        return []
    if 'notes' not in result:
        print(f"no facts, result: {result}")
        return []
    return result['notes']
