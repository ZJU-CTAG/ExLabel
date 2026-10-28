You are an expert in software testing and code evaluation.

You will be given:
- A `Problem statement`: a natural language description of the current software problem
- A `Test patch`: a code snippet that implements a test case for the problem
- A `Annotation`: 
  - `Reasonable`: a boolean indicating whether the test patch is reasonable
  - `Notes`: detailed notes explaining the judgment
- A `Previous model judgment`: the model's prior judgment on the reasonability of the test patch that represents how it successfully evaluated the test

Your task is to extract two key aspects from this successful judgment:
1. **perspective**: how this judgment thought about this issue - that is, how the problem was understood in a way that **led to its successful evaluation**. This should be abstract and not name specific code entities.
2. **experience**: abstract the reasoning behind this test judgment. What principle, pattern, or insight can be generalized from this judgment and applied to future judgments?

## Guidelines: 
1. Analyze the judgment and the issue, determine why this judgment could succeed in evaluating the test patch. 
2. Compare the the model judgment with the human annotation, extract insights on how to evaluate this kind of issue.

## Important Notes: 
  - Perspective should be at the level of thinking, not specific implementation details.
  - Perspective should be expressed in as generalized and abstract terms as possible.
  - Experience explains *why* judgment succeeded, in abstract and transferable terms.
  - You could extract *at most three* experiences.
  - Do not mention specific function names, variable names, or string contents from the actual code.

Your output must strictly follow the JSON format shown below:
{  
  "perspective": "<1-2 sentences to describe how this judgment understood this issue >", 
  "experience": "<1-3 concise principles or insights that can guide future judgments>" 
}