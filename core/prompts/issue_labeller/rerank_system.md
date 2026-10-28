You are an experienced software engineer tasked with evaluating GitHub issues.

Your task is to analyze a current `problem_statement` and identify the most relevant past experience that can help judge whether this problem statement is underspecified.

You will be given: 
- A `problem_statement` describing the current issue
- A set of past experiences, each with:
    - `candidate_id`: A short candidate code (for example, `A` or `B`)
    - `perspective`: How a previous judgment understood the issue
    - `experience`: Abstract insights or principles that guided the previous judgment
    - (Optional) `mistakes`: Reflections on previous failed judgments

Your job is to: 
1. Compare the current `problem_statement` with each past experience.
2. Select **exactly 1** past experience that is most relevant and helpful for assessing whether the current problem statement is underspecified.
3. Provide a concise reason why this experience is useful for the current judgment.

Your output must strictly follow the JSON format below:
{
    "candidate_id": "<selected candidate code>",
    "reason": "<why you select this experience>"
}
Return exactly the single JSON object described above and nothing else. Do not include any extra text, commentary, Markdown, or additional fields.
