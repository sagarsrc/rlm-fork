# RLM Podcast — Reference Sheet

30–40 min conversation. No strict timestamps, but the arc below keeps energy and clarity.

---

## Opening (first ~5 min)

- Hook: dynamic workflows are suddenly everywhere — Anthropic's Claude Code now does them, and it cites the RLM paper.
- Show the tweets:
  - Alex Zhang (`@a1zhang`): https://x.com/a1zhang/status/2060071701879066626
  - Omar Khattab (`@lateinteraction`): https://x.com/lateinteraction/status/2041961613721239745
- Mention Alex Zhang's point: Opus 4.8 + dynamic workflows in Claude Code looks like the first frontier model seriously trained as an RLM.
- Mention Omar Khattab's point: Anthropic's managed-agents blog cites RLM as solving compaction limitations — a frontier lab acknowledging academic work instead of reinventing it.
- Name-drop: RLM = Recursive Language Model.
- Frame: this paper is why frontier coding agents are moving toward programmatic, recursive sub-agent orchestration.
- Rough plan for the call:
  1. Tiny demo first.
  2. What problem it solves and why normal LLMs fail.
  3. How RLM architecture works.
  4. Bigger head-to-head demo.
  5. Sub-LLMs, root LLM, and `llm_query` in detail.
- Mention the repo on screen has a working tiny example.

---

## Part 1 — Tiny RLM demo + where each approach shines

### Show the tiny repo example
- Walk through the smallest RLM call in the repo.
- Show the context variable, the Python step, the truncated output, and `FINAL`.
- Keep it under a minute; just prove the loop exists.

### Side-by-side mental picture
- Normal LLM call: prompt goes into the model; model returns answer in one shot.
- RLM call: model writes Python; Python touches a huge context; model only sees a short summary; loop repeats until `FINAL`.

### Bring in "why RLMs?" early
- Simple LLM shines: short prompts, single reasoning step, answer fits in one response.
- RLM shines: long inputs, many small judgments, answer must be built up piece by piece.
- Key line: context window is not the only bottleneck; task complexity per token matters.

---

## Part 2 — The problem and the data

### OOLONG and OOLONG-Pairs
- OOLONG = benchmark with ~787 TREC questions.
- `trec_coarse` task: "which label is most common?" — linear, O(n).
- `oolong-pairs` task: "find pairs that share a topic but have conflicting answers" — quadratic, O(n²).
- 787 questions → ~309K unique pairs.

### Why O(n²) is hard
- A model cannot guess the full pair list from a single glance.
- It must compare question i against question j for almost every pair.
- Even a 32K-token version of this task breaks a top model.

### Where GPT-5 falls down
- GPT-5 on OOLONG linear: ~44% — okay.
- GPT-5 on OOLONG-Pairs: ~0.1% F1 — essentially zero.
- 32K tokens is far below GPT-5's 272K window, so the window is not the problem.
- The problem is the number of independent judgments.
- Contrast with a needle-in-haystack test: that is O(1), and models do fine at 1M tokens.

---

## Part 3 — Architecture at a high level

### LLM ↔ REPL loop
- Root LLM writes Python code.
- REPL executes it against a Python variable that can hold 10M+ tokens.
- REPL returns truncated output.
- Root LLM never sees the whole context, only summaries and tool outputs.
- Loop ends when root LLM calls `FINAL("answer")` or `FINAL_VAR(variable)`.

### RAG vs RLM
- RAG: human hardcodes chunking → chunks enter context → one answer.
- RLM: LLM writes its own chunking/filtering/aggregation code → adapts turn by turn.
- Three differences to hit:
  1. Who writes the pipeline — human vs model.
  2. Where the prompt lives — inside context window vs outside in a variable.
  3. Fixed one-shot vs interactive multi-turn loop.

---

## Part 4 — Bigger demo: 100 questions

### Setup
- Start with a tiny problem the audience can hold in their head.
- Then fire the same 100 questions at:
  - a plain LLM call, and
  - the RLM system.
- Explain that the 100 questions spawn many sub-LLM calls; this is expected and is the point.
- Let the demo breathe; it takes time.

### What to call out while it runs
- Plain LLM tries to answer in one context and misses nuance or hallucinates.
- RLM chunks the work, delegates each chunk, and aggregates.
- The root LLM is the planner; the sub-LLMs are the readers/workers.

---

## Part 5 — Root LLM, sub-LLM, and `llm_query`

### Definitions to land
- **Root LLM**: the brain. Writes all Python strategy. Decides when to probe, when to aggregate, when to stop.
- **Sub-LLM**: the reader tool. Receives a small chunk + a narrow question and returns a short answer.
- **REPL**: the executor. Holds the context variable, runs root LLM's code, calls `llm_query`.
- **`llm_query()`**: the bridge function. A normal Python function in the REPL that calls the sub-LLM API.

### One full turn
1. Root LLM sees truncated history.
2. Root LLM emits Python code with a plan.
3. REPL executes it — slices context, loops, calls `llm_query` many times.
4. Sub-answers accumulate in Python variables.
5. REPL prints a short summary back to root LLM.
6. Root LLM decides next step or calls `FINAL`.

### Optional: recursion
- `rlm_query()` can spawn a nested RLM with its own REPL and context.
- Useful for decomposing hard sub-problems. Falls back to `llm_query` at max depth.

---

## Closing

- Recap:
  - Simple LLM = great for short, single-step reasoning.
  - RLM = scales to long inputs and complex tasks by letting the model write code that delegates work.
- Open questions:
  - Cost can spiral if the root LLM keeps searching.
  - Guardrails and iteration budgets matter in practice.
- CTA: point viewers to the repo and paper.
- Closing hook: this is not a niche idea anymore — Claude Code's dynamic workflows are an early commercial form of exactly this design. RLM thinking is becoming the default for serious agents.

---

## Appendix — quick facts to pull from

- GPT-5 window: ~272K tokens.
- Qwen3-8B window: ~32K tokens.
- OOLONG: 787 questions, ~77K chars total.
- OOLONG-Pairs pairs: ~309K.
- GPT-5 F1 on OOLONG-Pairs: 0.1%.
- Key terms: context rot, O(n) vs O(n²), root LLM, sub-LLM, REPL, `llm_query`, `FINAL`.
