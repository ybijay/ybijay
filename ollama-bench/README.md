# ollama-bench

A small capability benchmark for local Ollama models: six short tasks, each
aimed at a different skill, so you can see which model is good at what.
Python 3.10+, standard library only.

```
python3 ollama_bench.py gemma4-32k:31b-it-q4_K_M qwen3-coder-64k:30b
python3 ollama_bench.py qwen3.8-64k:27b --tasks code,bug     # just some tasks
```

It runs every task on one model before moving to the next, so each model loads
only once. Expect a few minutes per model; longer for dense or "thinking" models.

## The tasks

| Task | Skill | Graded by |
|---|---|---|
| `code` | coding: implement a duration parser from a precise spec | runs the code against 19 hidden test cases, naming any that fail |
| `bug` | code review: find a planted off-by-one bug | `LINE: 8` |
| `math` | multi-step arithmetic with a timeline | `ANSWER: 29.8` |
| `schedule` | planning under constraints (the greedy plan is wrong) | `ANSWER: 10` |
| `format` | following a strict output format | exactly 3 lines, `- ` prefix, max 12 words |
| `design` | architecture: a broker sharing one GPU among users | **you** (word count only) |

The answer keys for `math` and `schedule` are verified by brute force in the
tests.

## Output

A line per task while it runs, then a summary (numbers below are illustrative):

```
TASK      SKILL                   gemma4-32k:31b-it-q4_K_M  qwen3-coder-64k:30b
code      coding                  PASS                      PASS
bug       code review             PASS                      PASS
math      multi-step arithmetic   PASS                      FAIL
...
score     auto-checked            5/5                       4/5
speed     tokens/s                9.8                       41.2
time      total                   412s                      96s
```

Every full answer is saved in `bench-results/<model>/<task>.md`, so you can
read *why* a model failed, and compare the `design` answers side by side.

## Fairness notes

- Uses each model's own sampling settings with a fixed seed. Forcing
  temperature 0 (`--temperature 0`) can make thinking models repeat themselves
  until they hit the token limit.
- Thinking is stripped before grading but saved in the answer file; thinking
  time still counts toward speed and total time.
- Answers are capped at 8192 tokens (`--num-predict`). A model that runs out
  before answering is reported as hitting the limit; read its Thinking section
  to see whether it was making progress or looping.
- Six tasks are a quick signal, not a leaderboard. One FAIL can be bad luck;
  a pattern (e.g. failing both reasoning tasks) is the real result.
- The `code` task runs the model's code on your machine (in a temp folder,
  10 s timeout). It's your own local model, but read it first if unsure.

## Tests

```
python3 -m unittest discover -s tests -v
```
