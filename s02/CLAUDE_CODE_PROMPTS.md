# Claude Code Prompts — Session 2

**QuickLoan | Multi-Turn Memory**

---

## How to use this sheet

These are prompts you type into Claude Code to add conversation memory to QuickLoan.
Good prompts are specific about four things:

```
1. Which file and function you are working on
2. What the inputs are
3. What the output must look like
4. Any constraints (names, types, error handling)
```

Open `s02/quickloan/` alongside this sheet.
There are 4 TODOs spread across 4 files. Fill them in order 1 → 4.

**What you are building:**
After this session, a customer can say "I earn Rs. 60,000 per month" in one
turn and ask "Am I eligible for a home loan?" in the next — and QuickLoan will use
the income figure without being told again.

---

## TODO 1 — Call load_dotenv() `quickloan/__init__.py`

Same as Session 1. The `__init__.py` is the first file Python runs when
the `quickloan` package loads — so the API key must be available here
before `config.py` or `tools.py` try to read it.

```
In quickloan/__init__.py, fill in the two steps at the TODO 1 block:

Step 1: Import the function
    from dotenv import load_dotenv

Step 2: Call it immediately after the import
    load_dotenv()
```

**Expected result:** two lines added. The rest of the file is already in place.

---

## TODO 2 — Add the history field to QuickLoanState `quickloan/state.py`

**What it does:** gives the graph a place to store the running conversation.
Every turn, `respond()` will read this list and append to it.

```
In quickloan/state.py, add one field to QuickLoanState below the
existing customer_message and response fields:

    history: list[dict]

Each dict in the list will have two keys:
    {"role": "user",      "content": "..."}
    {"role": "assistant", "content": "..."}

Do not add any other fields.
```

**Expected result:**

```python
class QuickLoanState(TypedDict):
    customer_message: str
    response:         str
    history:          list[dict]
```

---

## TODO 3 — Update respond() to use history `quickloan/nodes.py`

**What it does:** sends the full conversation to the LLM so it can
refer back to earlier turns. Then saves the new turn to history.

```
In quickloan/nodes.py, replace the TODO placeholder in respond() with
this three-step implementation:

Step A — Build the message list:

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    Loop over `history` and append each turn:
      - {"role": "user", ...}       → HumanMessage(content=turn["content"])
      - {"role": "assistant", ...}  → AIMessage(content=turn["content"])

    Then append the new question:
      messages.append(HumanMessage(content=state["customer_message"]))

Step B — Call the LLM:

    try:
        result = llm.invoke(messages)
        response_text = result.content
    except Exception as e:
        print(f"[QuickLoan] LLM error: {e}")
        response_text = "I am temporarily unavailable. Please try again in a moment."

Step C — Append this turn to history and return both fields:

    new_history = history + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": response_text},
    ]
    return {"response": response_text, "history": new_history}

Important: return BOTH "response" and "history" in the dict.
If you omit "history", the checkpointer has nothing new to save.
```

**Why AIMessage and HumanMessage?**
The LLM API expects message objects, not raw dicts. `history` stores plain
dicts so they are easy to inspect; this loop converts them for the LLM call.

---

## TODO 4 — Wire the checkpointer in build_graph() `quickloan/agent.py`

**What it does:** attaches a SQLite checkpointer so LangGraph saves state
between turns — even if the script restarts.

```
In quickloan/agent.py, uncomment the build_graph() function and delete
the two placeholder lines below it.

The function should:
1. Create the builder:   builder = StateGraph(QuickLoanState)
2. Add the respond node: builder.add_node("respond", respond)
3. Set entry point:      builder.set_entry_point("respond")
4. Add edge to END:      builder.add_edge("respond", END)
5. Use the passed-in checkpointer (or MemorySaver if none given):
       if checkpointer is None:
           checkpointer = MemorySaver()
6. Compile and return:   return builder.compile(checkpointer=checkpointer)

The thread_id and config in run() are already provided — do not change them.
```

**Why accept checkpointer as a parameter?**
Tests inject an in-memory `MemorySaver()` instead of writing to disk.
Production code calls `build_graph()` with `SqliteSaver` (already done in
`run()`). Same function, swappable persistence.

**Why NOT pass `history` to `graph.invoke()`?**
The checkpointer loads history automatically from the previous snapshot.
Passing `history: []` every time would reset memory on every turn.

---

## Running the tests

After completing all four TODOs:

```bash
# from the repo root (quickloan/)
pytest s02/tests/ -v
```

All tests should be green. If any fail, use the prompts below.

---

## Debugging prompts — when a test fails

**Test fails: `test_history_is_included_in_llm_messages`**
```
The test test_history_is_included_in_llm_messages is failing.
It patches llm.invoke and checks that the messages list passed to it
contains a HumanMessage and AIMessage from a prior history entry.
Look at my respond() in quickloan/nodes.py and check that the loop
over `history` appends HumanMessage for "user" and AIMessage for "assistant".
```

**Test fails: `test_respond_returns_updated_history`**
```
The test test_respond_returns_updated_history is failing.
It checks that the dict returned by respond() contains a "history" key
with the new turn appended.
Look at my respond() return statement and make sure I return
{"response": response_text, "history": new_history} — not just "response".
```

**Test fails: `test_same_thread_id_shares_history`**
```
The test test_same_thread_id_shares_history is failing.
It invokes the graph twice with the same thread_id and checks that the
second call sees the first turn in its history.
Look at my build_graph() in quickloan/agent.py and make sure I call
builder.compile(checkpointer=checkpointer) and return the result.
If build_graph() still raises NotImplementedError, the TODO 4 placeholder
was not removed.
```

**Test fails: `test_different_thread_ids_have_separate_histories`**
```
The test test_different_thread_ids_have_separate_histories is failing.
It invokes the graph with two different thread_ids and checks that they
do not share history.
This test only fails if the checkpointer is shared incorrectly or if
the thread_id is hardcoded. Confirm that build_graph() accepts a
checkpointer parameter and compiles with it.
```

---

## Understanding prompts

```
Explain in one sentence why we pass history to the LLM as message objects
(HumanMessage, AIMessage) rather than as plain strings.
```

```
Explain the difference between MemorySaver and SqliteSaver.
Which one loses data when the script restarts, and why?
```

```
Explain why graph.invoke() is called with {"customer_message": ..., "response": ""}
but NOT with a "history" key, even though history is part of QuickLoanState.
```

---

## Extension prompts — for fast finishers

**Show the conversation turn count:**
```
In quickloan/agent.py, after printing the QuickLoan response, also print
how many turns the conversation has had so far:
    print(f"  [History: {len(result['history'])} turns]")
```

**Limit history length:**
```
In quickloan/nodes.py, before building the message list, trim history
to the last 6 entries (3 turns) so the LLM context does not grow forever:
    history = history[-6:]
Keep the trimmed list when building messages but still append the full
new turn before returning.
```

**Inspect the checkpoints database:**

Run these from the repo root after a few turns:

```bash
# 1. What tables exist?
python -c "
import sqlite3
conn = sqlite3.connect('data/checkpoints.db')
print(conn.execute('SELECT name FROM sqlite_master WHERE type=\"table\"').fetchall())
"

# 2. Browse all rows (skips the binary blob — readable columns only)
python -c "
import sqlite3
conn = sqlite3.connect('data/checkpoints.db')
rows = conn.execute(
    'SELECT thread_id, checkpoint_id, parent_checkpoint_id, metadata FROM checkpoints'
).fetchall()
for r in rows: print(r)
"

# 3. How many checkpoints per conversation?
python -c "
import sqlite3
conn = sqlite3.connect('data/checkpoints.db')
rows = conn.execute(
    'SELECT thread_id, COUNT(*) FROM checkpoints GROUP BY thread_id'
).fetchall()
for r in rows: print(r)
"
```

You will see **3 rows per message turn** — one for input received, one for the routing step, one for the node completing. The `parent_checkpoint_id` column chains them together like a linked list.

Prefer a visual interface? Download [DB Browser for SQLite](https://sqlitebrowser.org/) (free, Mac/Windows/Linux) and open `data/checkpoints.db` directly.

---

## The principle

> **Return only what changed.**
>
> LangGraph merges the dict you return back into the full state.
> You only need to return the keys that changed.
>
> respond() in Session 1 returned `{"response": ...}` — one key.
> respond() in Session 2 returns `{"response": ..., "history": ...}` — two keys.
> It never returns `customer_message` because respond() did not change it.
>
> The same principle applies to every node you write in this course:
> return the minimum set of changed keys, not a copy of the whole state.
