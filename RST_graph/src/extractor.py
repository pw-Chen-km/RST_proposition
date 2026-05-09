import os
import re
import logging
from typing import Dict, Any, List, Tuple

from openai import AsyncOpenAI
from . import prompts as active_prompts
from .constants import NODE_ROLES, RELATION_FAMILIES

logger = logging.getLogger(__name__)

TUPLE_DELIMITER = "<|>"
COMPLETION_DELIMITER = "<|EOF|>"

# Node type markers (accept both old "entity" and new "NODE" for backward compatibility)
NODE_TYPE_MARKERS = {"entity", "NODE"}
REL_TYPE_MARKERS = {"relation", "REL"}

# Placeholder tokens that should NEVER appear as node names in relations
PLACEHOLDER_TOKENS = {"entity", "NODE", "source_node", "target_node", "relation_family", "REL"}


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


def _normalize_node_name(value: str) -> str:
    """Normalize node name for fuzzy matching."""
    return re.sub(r"\s+", " ", value.strip().rstrip(".")).lower()


def _is_placeholder(name: str) -> bool:
    """Check if the name is a known placeholder token."""
    normalized = name.strip().lower()
    if normalized in PLACEHOLDER_TOKENS:
        return True
    # Also catch entity1, entity_1, etc.
    if re.match(r'^entity[_\d]+$', normalized):
        return True
    return False


def parse_extracted_content_strict(content: str) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Strict parsing with validation.
    
    Returns: (valid_result, dropped_relations)
    where dropped_relations contains info about why each relation was dropped.
    """
    nodes = []
    valid_edges = []
    dropped_relations = []
    
    # Clean up completion delimiter
    content = content.split(COMPLETION_DELIMITER)[0].strip()
    
    # First pass: collect all valid nodes
    node_names_normalized = {}  # normalized -> original
    node_names_original = set()  # original case-sensitive
    
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(TUPLE_DELIMITER)
        parts = [p.strip().strip('"').strip("'") for p in parts]
        
        if len(parts) >= 3 and parts[0] in NODE_TYPE_MARKERS:
            name = parts[1]
            role = parts[2].strip() or "Unknown"
            
            # Skip if node name is a placeholder
            if _is_placeholder(name):
                logger.warning(f"Dropping node with placeholder name: {name}")
                continue
                
            nodes.append({
                "entity_name": name,
                "entity_type": role,
                "description": f"Proposition node: {name}"
            })
            node_names_original.add(name)
            node_names_normalized[_normalize_node_name(name)] = name
    
    # Second pass: validate relations
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(TUPLE_DELIMITER)
        parts = [p.strip().strip('"').strip("'") for p in parts]
        
        if len(parts) >= 4 and parts[0] in REL_TYPE_MARKERS:
            source = parts[1]
            target = parts[2]
            family = parts[3].upper()
            
            drop_reason = None
            
            # Check 1: source/target should not be placeholder tokens
            if _is_placeholder(source):
                drop_reason = f"source is placeholder: {source}"
            elif _is_placeholder(target):
                drop_reason = f"target is placeholder: {target}"
            # Check 2: source/target must exist in node list
            elif source not in node_names_original:
                # Try fuzzy match
                source_norm = _normalize_node_name(source)
                if source_norm in node_names_normalized:
                    source = node_names_normalized[source_norm]
                else:
                    drop_reason = f"source not in nodes: {source[:50]}..."
            elif target not in node_names_original:
                # Try fuzzy match
                target_norm = _normalize_node_name(target)
                if target_norm in node_names_normalized:
                    target = node_names_normalized[target_norm]
                else:
                    drop_reason = f"target not in nodes: {target[:50]}..."
            # Check 3: no self-loops
            elif source == target:
                drop_reason = f"self-loop: {source}"
            
            if drop_reason:
                dropped_relations.append({
                    "line": line,
                    "source": parts[1],
                    "target": parts[2],
                    "family": family,
                    "reason": drop_reason
                })
                logger.warning(f"Dropping relation: {drop_reason}")
                continue
            
            valid_edges.append({
                "source": source,
                "target": target,
                "keywords": family,
                "weight": 1.0,
                "description": f"Relation: {source} {family} {target}"
            })
    
    result = {"nodes": nodes, "edges": valid_edges}
    return result, dropped_relations


def parse_extracted_content(content: str) -> Dict[str, List[Dict[str, Any]]]:
    """Legacy wrapper for backward compatibility."""
    result, _ = parse_extracted_content_strict(content)
    return result


async def extract_propositions(
    text_chunk: str, 
    client: AsyncOpenAI, 
    model: str = None,
    language: str = "English",
    max_retries: int = 2
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Directly extracts proposition nodes and edges from text utilizing OpenAI.
    With strict validation and retry with ICL example on format errors.
    """
    model = model or os.getenv("PROPOSITION_EXTRACT_MODEL", "gpt-4o")
    system_prompt = _format_prompt(active_prompts.ENTITY_EXTRACTION_SYSTEM_PROMPT, language)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Extract the proposition graph from the following text:\n\n{text_chunk}"}
    ]
    
    for attempt in range(max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0
            )
            content = response.choices[0].message.content
            
            # Strict parsing with validation
            result, dropped = parse_extracted_content_strict(content)
            
            # Log extraction stats
            total_relations = len(result["edges"]) + len(dropped)
            if total_relations > 0:
                drop_rate = len(dropped) / total_relations * 100
                logger.info(f"Extraction attempt {attempt+1}: {len(result['nodes'])} nodes, "
                          f"{len(result['edges'])}/{total_relations} valid relations "
                          f"({drop_rate:.1f}% dropped)")
            
            # If we have invalid relations and haven't exhausted retries, try again with ICL
            if dropped and attempt < max_retries:
                logger.warning(f"Found {len(dropped)} invalid relations, retrying with ICL example...")
                
                # Build correction prompt with full ICL example
                correction_prompt = _build_retry_prompt_with_icl(dropped, result["nodes"])
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": correction_prompt})
                continue
            
            # Return result (best effort after retries or if no dropped relations)
            return result
            
        except Exception as e:
            logger.error(f"Extraction API call failed on attempt {attempt+1}: {e}")
            if attempt == max_retries:
                raise
    
    return result


def _build_retry_prompt_with_icl(dropped_relations: List[Dict], valid_nodes: List[Dict]) -> str:
    """
    Build a retry prompt with full ICL example.
    Asks model to re-output COMPLETE extraction with correct format.
    """
    
    # Group dropped reasons
    by_reason = {}
    for rel in dropped_relations:
        reason = rel["reason"]
        by_reason[reason] = by_reason.get(reason, 0) + 1
    
    # Build error summary
    error_summary = []
    for reason, count in sorted(by_reason.items(), key=lambda x: -x[1]):
        error_summary.append(f"  - {reason}: {count} case(s)")
    
    # Get sample invalid lines
    sample_errors = dropped_relations[:3]
    
    # Build ICL example
    icl_example = f"""
---EXAMPLE OF CORRECT OUTPUT FORMAT---

Given text: "The peninsula gradually rises toward the interior. The peninsula reaches a maximum height of more than 70 feet."

CORRECT output (nodes first, then relations):

NODE{TUPLE_DELIMITER}The peninsula gradually rises toward the interior.{TUPLE_DELIMITER}Context
NODE{TUPLE_DELIMITER}The peninsula reaches a maximum height of more than 70 feet.{TUPLE_DELIMITER}Evidence
REL{TUPLE_DELIMITER}The peninsula gradually rises toward the interior.{TUPLE_DELIMITER}The peninsula reaches a maximum height of more than 70 feet.{TUPLE_DELIMITER}SUPPORTS

CRITICAL RULES demonstrated above:
1. NODE lines: NODE{{delimiter}}full_proposition_text{{delimiter}}role
2. REL lines: REL{{delimiter}}SOURCE_PROPOSITION{{delimiter}}TARGET_PROPOSITION{{delimiter}}family
3. Source and target in REL must match NODE lines EXACTLY (same text)
4. NO placeholder words like 'entity', 'NODE', 'source_node' in source/target fields
5. NO node roles (Claim, Evidence, etc.) as relation family
"""
    
    lines = [
        "---FORMATTING ERRORS DETECTED---",
        "",
        "Your previous output had formatting errors that prevented correct graph construction:",
        "",
        "Error summary:"
    ]
    lines.extend(error_summary)
    
    lines.extend([
        "",
        "Sample problematic lines:"
    ])
    for rel in sample_errors:
        lines.append(f"  - {rel['line'][:120]}")
        lines.append(f"    Reason: {rel['reason']}")
    
    lines.extend([
        "",
        icl_example,
        "",
        "---YOUR TASK---",
        "",
        "Re-extract the ENTIRE graph from the original text with CORRECT FORMAT:",
        "",
        "1. Output ALL proposition nodes first (NODE format)",
        "2. Then output ALL valid relations (REL format)",
        "3. Source and target in REL must be EXACT copies of NODE proposition text",
        "4. DO NOT use placeholder words in source/target: entity, NODE, source_node, target_node",
        "5. Valid relation families: " + ", ".join(RELATION_FAMILIES),
        "6. If a relation source/target is unclear, SKIP that relation entirely",
        "",
        "Output your complete corrected extraction below:"
    ])
    
    return "\n".join(lines)
