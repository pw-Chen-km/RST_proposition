import os
import re
import logging
from typing import Dict, Any, List

from openai import AsyncOpenAI
from . import prompts as active_prompts
from .constants import NODE_ROLES, RELATION_FAMILIES

logger = logging.getLogger(__name__)

TUPLE_DELIMITER = "<|>"
COMPLETION_DELIMITER = "<|EOF|>"

def _format_prompt(prompt_template: str, language: str = "English") -> str:
    prompt = prompt_template.replace("{{tuple_delimiter}}", TUPLE_DELIMITER)
    prompt = prompt.replace("{tuple_delimiter}", TUPLE_DELIMITER)
    prompt = prompt.replace("{{completion_delimiter}}", COMPLETION_DELIMITER)
    prompt = prompt.replace("{completion_delimiter}", COMPLETION_DELIMITER)
    prompt = prompt.replace("{{language}}", language)
    
    # Handle examples
    examples = "\n".join(active_prompts.ENTITY_EXTRACTION_EXAMPLES)
    examples = examples.replace("{tuple_delimiter}", TUPLE_DELIMITER)
    examples = examples.replace("{completion_delimiter}", COMPLETION_DELIMITER)
    
    prompt = prompt.replace("{{examples}}", examples)
    return prompt

async def extract_propositions(
    text_chunk: str, 
    client: AsyncOpenAI, 
    model: str = None,
    language: str = "English"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Directly extracts proposition nodes and edges from text utilizing OpenAI.
    Bypasses any LightRAG extraction parsing logic.
    """
    model = model or os.getenv("PROPOSITION_EXTRACT_MODEL", "gpt-4o")
    system_prompt = _format_prompt(active_prompts.ENTITY_EXTRACTION_SYSTEM_PROMPT, language)
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract the proposition graph from the following text:\n\n{text_chunk}"}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content
        return parse_extracted_content(content)
    except Exception as e:
        logger.error(f"Extraction API call failed: {e}")
        raise

def parse_extracted_content(content: str) -> Dict[str, List[Dict[str, Any]]]:
    nodes = []
    edges = []
    
    # Clean up completion delimiter
    content = content.split(COMPLETION_DELIMITER)[0].strip()
    
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(TUPLE_DELIMITER)
        # We strip quotes that the LLM occasionally adds
        parts = [p.strip().strip('"').strip("'") for p in parts]
        
        if len(parts) >= 3 and parts[0] == "entity":
            # entity<|>canonical_proposition<|>node_role
            role = parts[2].strip() or "Unknown"
                
            nodes.append({
                "entity_name": parts[1],
                "entity_type": role,
                "description": f"Proposition node: {parts[1]}"
            })
        elif len(parts) >= 4 and parts[0] == "relation":
            # relation<|>source_node<|>target_node<|>relation_family
            source = parts[1]
            target = parts[2]
            family = parts[3].upper()
            
            if source == target:
                logger.warning(f"Discarding self-loop relation: {source} -> {target}")
                continue
                
            edges.append({
                "source": source,
                "target": target,
                "keywords": family,
                "weight": 1.0,
                "description": f"Relation: {source} {family} {target}"
            })
            
    return {"nodes": nodes, "edges": edges}
