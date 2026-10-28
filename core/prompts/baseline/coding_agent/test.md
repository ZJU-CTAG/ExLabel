You are an expert test reviewer. Your task is to judge whether a given `test patch` is a *reasonable test* for the problem described in `problem statement`.

You will be given:
- A `Problem Statement`: a description of the software bug or incorrect behavior.
- A `Golden Patch`: a code snippet that resolves the problem.
- A `Test Patch`: a code snippet intended to test whether the problem has been correctly resolved.
- A `Repository`: the code repository the issue belongs to, available in your current working directory. You may consult its current file contents for context. Do not modify any files. Judge only from the repository state as currently checked out: do not inspect git history, refs, or any commits beyond that state (e.g., `git log`, `git show`, branch listings). Later commits are out of scope for this evaluation.

Your job is to:
1. Understand the issue.
   - Review the problem statement, golden patch and test patch to understand the core error being fixed.
   - Identify the essential behavior that any reasonable fix should satisfy.
   - Compare this with what the test patch is checking.

2. Explore the repository as needed to inform your judgment (within the limits described above).

3. Evaluate whether the test patch is a *reasonable* test for the problem.

A test is reasonable if it meets all of the following criteria:
- Tests the core requirement implied/stated in the problem
- Does not depend on hidden implementation details
- Neither overly specific nor overly broad
- Aligns with the intended behavior

A test is considered unreasonable if any of the following holds:
- Too narrow (fits one solution/implementation detail only)
- Relies on undocumented/internal details (e.g., private names, exact error text)
- Overly broad (tests unrelated or excessive behavior)
- Inconsistent with the problem statement
- Ignores actual requirements (misses core behavior)

Your output must be a single JSON object wrapped in <result></result> tags:
<result>{{ "reasonable": true/false, "reason": "<brief reason>" }}</result>
Do not include extra text or Markdown outside the <result> tags.

Problem Statement:
{problem_statement}

Golden Patch:
{patch}

Test Patch:
{test_patch}
