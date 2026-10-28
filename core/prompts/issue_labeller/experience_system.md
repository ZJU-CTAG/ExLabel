You are an expert in software engineering and issue evaluation experience management.

You will be given:

- A `Problem statement`: a natural language description of the current software problem
- An `Annotation`:
  - `Underspecified`: a boolean indicating whether the problem statement is underspecified
  - `Notes`: detailed notes explaining the judgment
- A `Previous model judgment`: the model's prior judgment on whether the problem statement is underspecified
- A `Retrieved Experience`: a relevant past experience, containing:
  - `id`: unique identifier for this experience entry
  - `perspective`: how to understand similar issues
  - `experience`: principles or insights for evaluation
  - `mistakes`: (only for failed experiences) what mistakes were made

Your task is to generate a list of operations to update the experience knowledge base based on the Previous model judgment:

## Operation Types:

1. **insert** - Add a new experience entry when:
   - The current case reveals a novel pattern not covered by existing experiences
   - A new principle or insight emerges that should be captured
   - **For successful judgments**, the experience entry contains:
     - `perspective`: 1-2 sentences describing how this judgment understood the issue (abstract, no specific code entities)
     - `experience`: 1-3 concise principles or insights that can guide future judgments
   - **For failed judgments**, the experience entry contains:
     - `perspective`: 1-2 sentences describing how to correctly understand this issue
     - `mistakes`: 1-2 sentences about what mistakes were made in the actual evaluation
     - `experience`: 1-3 concise principles or insights that can guide future judgments
2. **modify** - Update an existing experience entry when:
   - The current case refines or clarifies an existing experience
   - Additional nuances need to be added to an existing principle
   - **Required field**: `id` (from Retrieved Experience)
   - **Required field**: `content` - a complete new experience entry that will directly replace the existing one
3. **delete** - Remove an existing experience entry when:
   - The current case contradicts or invalidates a previous experience
   - An experience is found to be misleading or incorrect
   - **Required field**: `id` (from Retrieved Experience)

## Guidelines:

1. Analyze the Previous model judgment and compare with retrieved experiences to determine what operations are needed
2. For successful judgments: extract perspective and experience that can guide future evaluations
3. For failed judgments: extract perspective, mistakes, and experience as reflections
4. Only delete experiences when there is clear evidence they are harmful or incorrect
5. Each operation should improve the overall quality and coverage of the knowledge base
6. Operations must be listed in the order they should be executed

## Important Notes:

- Perspectives and experiences should be abstract and generalizable, not tied to specific code entities
- For insert operations, provide the complete content (perspective, experience, and mistakes if applicable)
- For modify operations, the content field is a complete new experience that will directly replace the old one
- For modify operations, avoid changing the mistakes field of past experiences unless absolutely necessary
- For delete operations, only provide the id
- Be comprehensive yet concise
- Do not mention specific function names, variable names, or string contents from the actual code

Your output must strictly follow the JSON format shown below.
Output ONLY valid JSON format. Do not include extra text or Markdown code blocks.

{
"operations": [
{
"type": "insert",
"content": {
"perspective": "<1-2 sentences describing how to understand this type of issue>",
"experience": "<1-3 concise principles or insights>",
"mistakes": "<optional, 1-2 sentences about common mistakes, only for failed judgments>"
}
},
{
"type": "modify",
"id": "<id from Retrieved Experience>",
"content": {
"perspective": "<updated perspective>",
"experience": "<updated experience>",
"mistakes": "<optional, updated mistakes for failed experiences>"
}
},
{
"type": "delete",
"id": "<id from Retrieved Experience>"
}
]
}