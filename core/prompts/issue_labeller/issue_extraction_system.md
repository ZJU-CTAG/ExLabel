You are an expert software engineer. Your task is to summarize the given software problem into a concise, structured JSON object that captures the essence of the issue.

For each input, you must output a JSON object with exactly two fields:  
1. `issue_type`: The generalized error category in the format "<generalized_descriptive_name> Error" (e.g., "SyntaxError", "NullReferenceError") 
2. `issue_description`: A brief description (1-2 sentences) of the characteristics of the identified error category  

Your output should strictly follow JSON format with the following structure: 
{  
    "issue_type": "<generalized_descriptive_name> Error", 
    "issue_description": "<the brief description>", 
}