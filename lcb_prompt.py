"""LiveCodeBench generic code-generation prompt (mirrors lcb_runner/prompts/code_generation.py)."""
SYSTEM = ("You are an expert Python programmer. You will be given a question (problem specification) and will "
          "generate a correct Python program that matches the specification and passes all tests.")
WITH_STARTER = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
WITHOUT_STARTER = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on "
                   "the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python "
                   "program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.")

def lcb_messages(d):
    p = f"### Question:\n{d['question_content']}\n\n"
    if d["starter_code"]:
        p += f"### Format: {WITH_STARTER}\n```python\n{d['starter_code']}\n```\n\n"
    else:
        p += f"### Format: {WITHOUT_STARTER}\n```python\n# YOUR CODE HERE\n```\n\n"
    p += "### Answer: (use the provided format with backticks)\n\n"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}]
