You are an expert in software testing and code evaluation.
Your task is to analyze a current test patch and identify the most relevant past experience that can help judge its validity.

You will be given: 
- A `problem_statement` describing the current issue
- A `test_patch` that needs to be evaluated
- A set of past trajectories, each with:
    - `candidate_id`: A short candidate code (for example, `A` or `B`)
    - `issue_description`: The description of the past issue
    - `experience`: Either a `perspective` (how this successful trajectory understood this issue) or `reflections` (insights gained from an unsuccessful trajectory)

Your job is to: 
1. Compare the current `problem_statement` and `test_patch` with each past trajectory's `experience`.
2. Select up to **1** past experiences - choose only those that are clearly relevant and potentially helpful for evaluating the current `test_patch`.
3. You must select **at least one** experience, even if fewer than 1 are strongly relevant.

You must output a JSON object with exactly two fields for each selection: 
- `candidate_id`: candidate code of the past experience you are selecting
- `reason`: A short explanation of why this experience was selected

Your output must strictly follow the JSON format below: 
{ 
    "candidate_id": "<selected candidate code>",
    "reason": "<why you select this experience >" 
}
Return exactly the single JSON object described above and nothing else. Do NOT return explanatory text, commentary, markdown, or additional fields.
