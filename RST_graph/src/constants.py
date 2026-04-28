# Custom Ontology Definitions
NODE_ROLES = [
    "Claim",
    "Evidence",
    "Cause",
    "Action",
    "Condition",
    "Context",
    "Outcome"
]

RELATION_FAMILIES = [
    "SUPPORTS",
    "CAUSES",
    "CONDITIONED_ON",
    "CONSTRAINS",
    "EXPLAINS",
    "LEADS_TO",
    "CONTRASTS_WITH"
]

# We overwrite the default entity types from LightRAG with our Operational Node Roles
OPERATIONAL_ENTITY_TYPES = NODE_ROLES
