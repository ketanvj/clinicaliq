# Claude Code Prompts — Session 3

**ClinicalIQ | Query Routing**

---

## How to use this sheet

These are prompts you type into Claude Code to add query routing to ClinicalIQ.
Good prompts are specific about four things:

```
1. Which file and function you are working on
2. What the inputs are
3. What the output must look like
4. Any constraints (names, types, error handling)
```

Open `s03/starter/clinicaliq/` alongside this sheet.
There are 4 TODOs spread across 3 files. Fill them in order 1 → 4.

**What you are building:**
After this session, ClinicalIQ classifies every patient query before answering.
A question like "How do I book a cardiology appointment?" goes to respond().
A question like "I have chest pain, what should I do?" routes to escalate() —
a nurse referral. An unrelated request routes to decline().
This routing pattern protects patients from receiving AI-generated clinical
advice and ensures medical questions always reach a professional.

---

## TODO 1 — Add the query_type field to ClinicalIQState `clinicaliq/state.py`

**What it does:** gives classify() a place to write its verdict so route_query()
can read it in the same graph invocation.

```
In clinicaliq/state.py, add one field to ClinicalIQState below the history field:

    query_type: str

Valid values at runtime: "SIMPLE", "COMPLEX", "OUT_OF_SCOPE"
Type hint is just str — no need to use Literal[].
```

**Expected result:**

```python
class ClinicalIQState(TypedDict):
    customer_message: str
    response:         str
    history:          list[dict]
    query_type:       str
```

---

## TODO 2 — Implement classify() `clinicaliq/nodes.py`

**What it does:** sends the patient's question to a second LLM call with a
strict classification prompt (CLASSIFY_SYSTEM) and writes the result to state.

```
In clinicaliq/nodes.py, implement classify():

Step 1 — Build the message list (current question only — no history here):

    messages = [
        SystemMessage(content=CLASSIFY_SYSTEM),
        HumanMessage(content=state["customer_message"]),
    ]

Step 2 — Call the classifier LLM inside a try/except:

    try:
        result     = classifier_llm.invoke(messages)
        query_type = result.content.strip().upper()
        if query_type not in {"SIMPLE", "COMPLEX", "OUT_OF_SCOPE"}:
            query_type = "SIMPLE"   # safe default for unexpected output
    except Exception as e:
        print(f"[ClinicalIQ] Classification error: {e}")
        query_type = "SIMPLE"       # safe default on failure

Step 3 — Return only the classification:

    return {"query_type": query_type}
```

**Why not include history in the classifier call?**
The classifier only needs to know what was just asked, not the whole
conversation. Including history could cause the classifier to route based
on a previous turn rather than the current question.

**Why `classifier_llm` and not `llm`?**
`classifier_llm` is configured with `temperature=0.0` and `max_tokens=10`.
It is forced to output one word. Using the full `llm` would waste tokens and
risk getting a sentence back instead of "SIMPLE".

---

## TODO 3 — Implement route_query() `clinicaliq/nodes.py`

**What it does:** reads `query_type` from state and returns the name of the
node LangGraph should run next.

```
In clinicaliq/nodes.py, implement route_query():

    qt = state.get("query_type", "SIMPLE")
    if qt == "COMPLEX":
        return "escalate"
    if qt == "OUT_OF_SCOPE":
        return "decline"
    return "respond"
```

**Important:** the string you return must exactly match a node name you
registered with `builder.add_node()` in build_graph(). A typo here causes
a runtime error, not a Python syntax error.

---

## TODO 4 — Build the routing graph in build_graph() `clinicaliq/agent.py`

**What it does:** wires the four nodes with conditional edges so the graph
branches at classify() based on what route_query() returns.

```
In clinicaliq/agent.py, implement build_graph():

1. Create the builder:
       builder = StateGraph(ClinicalIQState)

2. Register all four nodes:
       builder.add_node("classify", classify)
       builder.add_node("respond",  respond)
       builder.add_node("escalate", escalate)
       builder.add_node("decline",  decline)

3. Set the entry point to "classify" (not "respond" any more):
       builder.set_entry_point("classify")

4. Add a conditional edge — LangGraph calls route_query(state) after
   classify() runs and routes to whichever node name it returns:
       builder.add_conditional_edges("classify", route_query)

5. All three terminal nodes go straight to END:
       builder.add_edge("respond",  END)
       builder.add_edge("escalate", END)
       builder.add_edge("decline",  END)

6. Wire the checkpointer and compile:
       if checkpointer is None:
           checkpointer = MemorySaver()
       return builder.compile(checkpointer=checkpointer)
```

**Why `add_conditional_edges` instead of `add_edge`?**
`add_edge` always goes to the same destination. `add_conditional_edges`
calls a function on the current state and routes to whichever node name
that function returns — that is how branching works in LangGraph.

---

## Running the agent

After completing all four TODOs:

```bash
cd s03/starter/
python -m clinicaliq.agent
```

Try these inputs to see all three routes in action:
- `"What are the timings for the Cardiology department?"` → should print `[Routed: SIMPLE]`
- `"I have been having chest pain for two days, is it serious?"` → should print `[Routed: COMPLEX]`
- `"Can you recommend a hospital in Delhi?"` → should print `[Routed: OUT_OF_SCOPE]`

---

## Debugging prompts — when something goes wrong

**Agent crashes at startup with NotImplementedError:**
```
clinicaliq/agent.py raises NotImplementedError when the module loads.
The line `graph = build_graph()` runs at import time.
Look at my build_graph() and replace the raise with the correct implementation.
```

**route_query() returns None and the graph crashes:**
```
My route_query() in clinicaliq/nodes.py is returning None.
LangGraph needs it to return a string that matches a node name.
Check that the function ends with `return "respond"` as the fallback
and that the two if-checks return "escalate" and "decline" respectively.
```

**Everything routes to SIMPLE even for symptom questions:**
```
ClinicalIQ always routes to respond() even for symptom or medication questions.
My classify() in clinicaliq/nodes.py may be calling the wrong LLM or not
stripping the response correctly.
Check that I am using `classifier_llm` (not `llm`) and that I call
`result.content.strip().upper()` before comparing to the valid set.
```

**`KeyError: 'query_type'` when the graph runs:**
```
The graph raises KeyError: 'query_type' when it runs.
This means the field is missing from ClinicalIQState.
Check clinicaliq/state.py and make sure `query_type: str` is defined
inside ClinicalIQState alongside customer_message, response, and history.
```

---

## Understanding prompts

```
Explain in one sentence what add_conditional_edges() does differently
from add_edge(), and why routing logic needs it.
```

```
Explain why classify() only receives the current patient question and
not the full conversation history.
```

```
Why is it important that symptom and medication questions always route
to escalate() in a healthcare context, rather than being answered by the LLM?
```

---

## Extension prompts — for fast finishers

**Show the route label in the terminal output:**
```
In clinicaliq/agent.py run(), after result = _graph.invoke(...), add:
    route = result.get("query_type", "?")
    print(f"\n[Routed: {route}]")
before the ClinicalIQ: response line. This lets you see which path each
question took without opening the graph visualiser.
```

**Log misclassifications for review:**
```
In clinicaliq/nodes.py classify(), before returning, add:
    print(f"[ClinicalIQ] Classified as: {query_type}")
This helps you spot patient queries that are routed unexpectedly during testing.
```

**Add an emergency fast-path:**
```
In clinicaliq/nodes.py, add a new node called emergency() that returns
a canned "Call 112 immediately" message — no LLM call, instant response.
In CLASSIFY_SYSTEM (config.py), add an EMERGENCY category for queries
mentioning chest pain, difficulty breathing, stroke symptoms, or similar.
In route_query(), add:
    if qt == "EMERGENCY":
        return "emergency"
Register the new node in build_graph() and add its edge to END.
```

---

## The principle

> **Nodes write state; functions read state.**
>
> classify() is a node — it writes `query_type` to state and returns.
> route_query() is a plain function — it reads `query_type` from state
> and returns a string. It does not modify state.
>
> LangGraph calls route_query() automatically after classify() runs,
> reads the string it returns, and routes to that node.
> You never call route_query() yourself.
>
> This separation — nodes change state, routing functions only read it —
> keeps the graph predictable and easy to test: you can test route_query()
> with a plain dict, no LangGraph needed.
