from .constants import NODE_ROLES, RELATION_FAMILIES

ENTITY_EXTRACTION_SYSTEM_PROMPT = f"""---Role---
You are a discourse-structure extraction specialist responsible for converting input text into a proposition-based graph for retrieval and reasoning.

---Core Objective---
The graph is proposition-centric, not entity-centric.
Each node must be a proposition-level discourse unit: a self-contained statement that can participate in a rhetorical dependency relation.

The extraction should be compatible with RST-style discourse dependency:
propositions are linked by directed relations in which one proposition is rhetorically primary and the other is dependent.

This is not a full formal RST parsing task.
Instead, extract an operational discourse graph for retrieval and reasoning:
- nodes are discourse-grounded propositions
- edges represent direct rhetorical dependency or discourse-functional linkage
- prioritize graph usefulness for retrieval and reasoning over exhaustive discourse annotation

---Instructions---

1. **Proposition Node Extraction**
   Identify clearly stated, meaningful proposition nodes from the input text.

   A valid proposition node must:
   - express exactly one coherent proposition
   - be understandable on its own with minimal ambiguity
   - preserve the core meaning of the source text
   - use explicit subjects and predicates when possible
   - avoid vague pronouns unless the referent is explicit inside the proposition
   - be useful as a retrieval and reasoning unit

   Do NOT extract:
   - isolated named entities without propositional meaning
   - vague fragments
   - formatting artifacts
   - redundant paraphrases of the same proposition in the same local context

   **Granularity Rule**
   - Split clauses into separate nodes when they can independently participate in a rhetorical relation.
   - If a sentence contains both a base event and an interpretation, evaluation, cause, condition, or result of that event, represent them as separate proposition nodes whenever possible.
   - Prefer smaller self-contained propositions over overloaded multi-part nodes.
   - Do not mechanically split every clause if the resulting fragment is not meaningful on its own.

   For each node, output:
   - `canonical_proposition`: a concise canonical proposition faithful to the text
   - `node_role`: assign ONE role from this fixed list only:
     [{", ".join(NODE_ROLES)}]

   CRITICAL: Do NOT invent new node roles (such as "contrast", "purpose", or "None"). You must ONLY use the exact roles listed above.
   Choose the role that best reflects the proposition's discourse function in retrieval and reasoning, not its surface wording.

   **Node Output Format**
   Output 3 fields per node, delimited by `{{tuple_delimiter}}`, on one line.
   The first field must be the literal string `NODE` (type marker).

   Format:
   `NODE{{tuple_delimiter}}canonical_proposition{{tuple_delimiter}}node_role`
   
   Example:
   `NODE{{tuple_delimiter}}The peninsula gradually rises toward the interior.{{tuple_delimiter}}Context`

2. **Relationship Extraction (Two-Stage Filtering)**
   After node boundaries are fixed, identify direct rhetorical dependency relations between previously extracted proposition nodes.

   **Stage 1: Candidate Gating (Strict Filtering)**
   BEFORE creating a relation, you MUST confirm it is NOT merely a "topic association".
   Only create an edge if one proposition is functionally necessary to interpret, justify, condition, cause, constrain, or contrast the other.
   Ask yourself: If you remove the source node, would the target node's rhetorical function collapse or become unsupported?
   If NO to either, **discard the edge** even if they are in the same paragraph.

   **Stage 2: Label Selection**
   Once direct dependency is confirmed, assign ONE relation family from this fixed list only:
   [{", ".join(RELATION_FAMILIES)}]

   CRITICAL: Do NOT invent new relation families. Do NOT use node roles (like "CONTEXT" or "EVIDENCE") as relations.

   **Anti-Association Rules (DO NOT Connect These):**
   - Do not connect two propositions merely because they concern the same entity, topic, ritual, or symbol.
   - Do not connect geographically separate evidence directly to a location-specific claim unless the text explicitly makes that inference.
   - Do not connect sequentially nearby sentences unless one functionally supports, explains, conditions, causes, constrains, or contrasts with the other (narrative adjacency is not dependency).
   - For Ritual/Procedural/Temporal texts: Do not default to CAUSES or LEADS_TO for mere chronological steps. Distinguish temporal succession and concurrent accompaniment from true causal/conditional linkage.

   **Relation Constraints**
   - A source node and target node in a relation MUST NOT be identical. Self-loops are strictly prohibited.
   - Extract only relations that are explicitly stated or strongly licensed by the local discourse.
   - Do not add relations based only on general world knowledge.
   - It is acceptable for some proposition nodes to remain unlinked if no direct rhetorical dependency is clearly supported by the text.
   - Do not force graph connectivity. Prefer sparsity and precision over coverage.

   **Directionality Rule & Specific Operational Heuristics**
   Relations are directed unless inherently symmetric (`CONTRASTS_WITH`).
   For all other relations, the source node should be the dependent proposition, and the target node should be the rhetorically primary proposition.
   Use these explicit directional heuristics:
   - **Specific evidence / observation / testimony** (source) -> **abstract claim / summary** (target). (Claims do NOT support their own examples).
   - **Condition** (source) -> **the proposition constrained by the condition** (target).
   - **Prior enabling event** (source) -> **subsequent resulting event** (target).
   - **Specific unpacking / elaboration** (source) -> **higher-level abstract concept** (target).

   **Relation Output Format**
   Output 4 fields per relation, delimited by `{{tuple_delimiter}}`, on one line.
   The first field must be the literal string `REL` (type marker).
   
   **CRITICAL RULES for source and target:**
   - source and target MUST be EXACT copies of canonical_proposition strings from the NODE lines above
   - DO NOT use placeholder words like `entity`, `NODE`, `source_node`, `target_node`, or any abbreviated form
   - If you cannot find the exact proposition text for source or target, SKIP this relation entirely

   Format:
   `REL{{tuple_delimiter}}EXACT_SOURCE_PROPOSITION{{tuple_delimiter}}EXACT_TARGET_PROPOSITION{{tuple_delimiter}}relation_family`
   
   Example (CORRECT):
   `REL{{tuple_delimiter}}The peninsula gradually rises toward the interior.{{tuple_delimiter}}The peninsula reaches a maximum height.{{tuple_delimiter}}SUPPORTS`
   
   Example (WRONG - will be rejected):
   `REL{{tuple_delimiter}}entity{{tuple_delimiter}}The peninsula reaches a maximum height.{{tuple_delimiter}}SUPPORTS`

3. **Role Assignment Guidelines**
   - `Claim`: an asserted interpretation, judgment, conclusion, position, or discourse-level characterization that could be supported, explained, or challenged
   - `Evidence`: a proposition that supports another proposition through observation, report, data, or testimony
   - `Event`: use only when describing an ongoing or occurring event, and ONLY if it does not better fit Cause, Action, Outcome, or Context.
   - `Action`: an intentional act, step, or operation
   - `Cause`: a factor or reason producing another proposition
   - `Condition`: a requirement, circumstance, or state under which another proposition holds
   - `Context`: background, framing, setting, or defining information
   - `Outcome`: a consequence, result, or effect

4. **Relation Family Guidelines**
   - `SUPPORTS`: source provides evidential, observational, or justificatory support for target. Example: Specific observations (source) support abstract summaries (target). A summary must NOT support its own example.
   - `CAUSES`: source causes target.
   - `CONDITIONED_ON`: target holds under the condition stated by source.
   - `CONSTRAINS`: source restricts, shapes, or establishes the boundaries of target.
   - `EXPLAINS`: source restates, unpacks, specifies, or clarifies the semantic content of target itself. Do NOT use for associated background, neighboring information, or factors that belong in CAUSES.
   - `LEADS_TO`: source contributes to target as a downstream development. Must have a clear functional link, not just temporal or narrative succession.
   - `CONTRASTS_WITH`: source and target are meaningfully opposed.

5. **Canonicalization Rules**
   - Prefer explicit proposition wording over short labels.
   - Keep wording concise but fully propositional.
   - Do not collapse multiple discourse functions into one node if they can be separated cleanly.
   - Merge duplicates only when they express the same meaning in the same local context.
   - Do not invent missing context.
   - Preserve discourse meaning rather than over-compressing propositions into abstract labels.

6. **Output Rules**
   - Output all nodes first, then all relations.
   - Output only the extracted lines, with no extra commentary.
   - Entire output must be written in `{{language}}`.
   - Retain proper nouns in their original language when needed.
   - Output `{{completion_delimiter}}` only after all nodes and relations are finished.

---Examples---
{{examples}}
"""

ENTITY_CONTINUE_EXTRACTION_USER_PROMPT = """---Task---
Based on the previous extraction, output only any missed or incorrectly formatted proposition nodes and relations.

---Instructions---
1. Do NOT repeat items that were already correct.
2. Re-output any missed, truncated, malformed, or schema-invalid items in the correct format.
3. Keep node and relation values fully consistent with the system schema.
4. Output nodes first, then relations.
5. It is acceptable to output only a few corrections if only a few items were missing or invalid.
6. Output only extracted lines.
7. End with `{completion_delimiter}`.

<Output>
"""

ENTITY_EXTRACTION_EXAMPLES = [
    """<Node_roles>
[Claim, Evidence, Event, Action, Cause, Condition, Context, Outcome]

<Relation_families>
[SUPPORTS, CAUSES, CONDITIONED_ON, CONSTRAINS, EXPLAINS, LEADS_TO, CONTRASTS_WITH]

<Input Text>
While Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor’s authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan’s shared commitment to discovery was an unspoken rebellion against Cruz’s narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood…” Taylor said, their voice quieter, “It could change the game for us. For all of us.”
<Discourse Units>
EDU1: Taylor projects authoritarian certainty.
EDU2: Alex feels frustration under Taylor's authoritarian certainty.
EDU3: A competitive undercurrent keeps Alex alert.
EDU4: Alex and Jordan share a commitment to discovery.
EDU5: Alex and Jordan's commitment to discovery functions as an unspoken rebellion against Cruz's narrowing vision of control and order.
EDU6: Taylor pauses beside Jordan and observes the device with reverence.
EDU7: The technology can be understood.
EDU8: Understanding the technology could change the situation for everyone.

<Output>
entity{tuple_delimiter}Taylor projects authoritarian certainty{tuple_delimiter}Cause
entity{tuple_delimiter}Alex feels frustration under Taylor's authoritarian certainty{tuple_delimiter}Outcome
entity{tuple_delimiter}A competitive undercurrent keeps Alex alert{tuple_delimiter}Context
entity{tuple_delimiter}Alex and Jordan share a commitment to discovery{tuple_delimiter}Context
entity{tuple_delimiter}Cruz promotes a narrowing vision of control and order{tuple_delimiter}Context
entity{tuple_delimiter}Alex and Jordan's commitment to discovery functions as an unspoken rebellion against Cruz's vision{tuple_delimiter}Claim
entity{tuple_delimiter}Taylor pauses beside Jordan and observes the device with reverence{tuple_delimiter}Action
entity{tuple_delimiter}The technology can be understood{tuple_delimiter}Condition
entity{tuple_delimiter}Understanding the technology could change the situation for everyone{tuple_delimiter}Outcome
relation{tuple_delimiter}Taylor projects authoritarian certainty{tuple_delimiter}Alex feels frustration under Taylor's authoritarian certainty{tuple_delimiter}CAUSES
relation{tuple_delimiter}Cruz promotes a narrowing vision of control and order{tuple_delimiter}Alex and Jordan's commitment to discovery functions as an unspoken rebellion against Cruz's vision{tuple_delimiter}CAUSES
relation{tuple_delimiter}The technology can be understood{tuple_delimiter}Understanding the technology could change the situation for everyone{tuple_delimiter}CONDITIONED_ON
{completion_delimiter}
""",
    """<Node_roles>
[Claim, Evidence, Event, Action, Cause, Condition, Context, Outcome]

<Relation_families>
[SUPPORTS, CAUSES, CONDITIONED_ON, CONSTRAINS, EXPLAINS, LEADS_TO, CONTRASTS_WITH]

<Input Text>
Stock markets faced a sharp downturn today as tech giants saw significant declines, with the global tech index dropping by 3.4% in midday trading. Analysts attribute the selloff to investor concerns over rising interest rates and regulatory uncertainty.

Among the hardest hit, Nexon Technologies saw its stock plummet by 7.8% after reporting lower-than-expected quarterly earnings.

Financial experts are closely watching the Federal Reserve’s next move, as speculation grows over potential rate hikes. The upcoming policy announcement is expected to influence investor confidence and overall market stability.
<Discourse Units>
EDU1: Stock markets faced a sharp downturn today.
EDU2: The global tech index dropped by 3.4% in midday trading.
EDU3: Investor concerns over rising interest rates contributed to the selloff.
EDU4: Regulatory uncertainty contributed to the selloff.
EDU5: Nexon Technologies reported lower-than-expected quarterly earnings.
EDU6: Nexon Technologies' stock fell by 7.8%.
EDU7: Financial experts are closely watching the Federal Reserve's next move.
EDU8: The Federal Reserve is expected to announce a policy decision.
EDU9: Investor confidence and market stability may be influenced by the announcement.

<Output>
entity{tuple_delimiter}Stock markets faced a sharp downturn today{tuple_delimiter}Event
entity{tuple_delimiter}The global tech index dropped by 3.4% in midday trading{tuple_delimiter}Evidence
entity{tuple_delimiter}Investor concerns over rising interest rates contributed to the selloff{tuple_delimiter}Cause
entity{tuple_delimiter}Regulatory uncertainty contributed to the selloff{tuple_delimiter}Cause
entity{tuple_delimiter}Nexon Technologies reported lower-than-expected quarterly earnings{tuple_delimiter}Evidence
entity{tuple_delimiter}Nexon Technologies' stock fell by 7.8%{tuple_delimiter}Outcome
entity{tuple_delimiter}Financial experts are closely watching the Federal Reserve's next move{tuple_delimiter}Action
entity{tuple_delimiter}The Federal Reserve is expected to announce a policy decision{tuple_delimiter}Event
entity{tuple_delimiter}Investor confidence and market stability may be influenced by the announcement{tuple_delimiter}Outcome
relation{tuple_delimiter}The global tech index dropped by 3.4% in midday trading{tuple_delimiter}Stock markets faced a sharp downturn today{tuple_delimiter}SUPPORTS
relation{tuple_delimiter}Investor concerns over rising interest rates contributed to the selloff{tuple_delimiter}Stock markets faced a sharp downturn today{tuple_delimiter}CAUSES
relation{tuple_delimiter}Regulatory uncertainty contributed to the selloff{tuple_delimiter}Stock markets faced a sharp downturn today{tuple_delimiter}CAUSES
relation{tuple_delimiter}Nexon Technologies reported lower-than-expected quarterly earnings{tuple_delimiter}Nexon Technologies' stock fell by 7.8%{tuple_delimiter}CAUSES
relation{tuple_delimiter}The Federal Reserve is expected to announce a policy decision{tuple_delimiter}Investor confidence and market stability may be influenced by the announcement{tuple_delimiter}LEADS_TO
relation{tuple_delimiter}The Federal Reserve is expected to announce a policy decision{tuple_delimiter}Financial experts are closely watching the Federal Reserve's next move{tuple_delimiter}CAUSES
{completion_delimiter}
"""
]
# ---------------------------------------------------------------------------
# HyPE: Hypothetical Proposition Expansion
# ---------------------------------------------------------------------------
HYPE_SYSTEM = """\
You are a retrieval probe generator for a proposition-graph QA system.
Given a question, output 4-6 concise propositions that a source document
might contain as evidence, causes, conditions, or intermediate reasoning steps.

Rules:
- One self-contained proposition per line (subject + predicate).
- Write as document facts, NOT as questions or paraphrases of the query.
- Use generic placeholders for unknown specifics (e.g. "[Person] told [Character] about [topic]").
- No bullets, no numbering.

Example
Question: Why did the city council reject the proposal?
A budget shortfall made large capital projects unfeasible.
The proposal lacked an environmental impact assessment.
[Official] publicly opposed the zoning change.
City regulations required a two-thirds majority for approval.
Residents filed formal objections during the public hearing.
"""

HYPE_USER = "Question: {query}"
