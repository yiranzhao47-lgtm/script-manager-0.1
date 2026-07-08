"""
Token budget estimator (dry-run mode).

Scans all aligned JSON files and meta.json, estimates input token counts
per episode using tiktoken, and prints a full budget report before any
LLM API call is made.  Helps the user decide on model choice and cost.
"""
# Implementation: Task 5 (Execution Layer)
