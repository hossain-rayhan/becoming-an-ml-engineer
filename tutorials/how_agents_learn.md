# How AI Agents Get Smarter — Frozen Models, RAG & Agent Kits

> **Tutorial format:** Each section below = **one slide**.
> Each slide has: a **Title**, a few **big on-slide bullets** (≤4 words — minimal text!),
> a **Visual** (diagram-first), an **Animate** cue (how it builds), and **Speaker notes** (what you say).
> Rule of thumb: **the diagram carries the slide, the bullets are just anchors, you say the details.**

---

## 🎨 Slide Design Guidelines (apply to every slide)

- **Background:** full **black** (`#000000`)
- **Title:** **bold**, **dark-orange** (`#CC5500` / `#D2691E`)
- **Body text:** **all bold**, mostly **white** (`#FFFFFF`)
- **Accents (optional):** use the same dark-orange sparingly for emphasis/keywords
- **Font:** clean sans-serif (e.g. Inter, Montserrat, Segoe UI); large sizes for readability
- **Contrast:** keep high — white/orange on black; avoid low-contrast greys
- **Motion:** **animate every diagram** — build it one element at a time, never dump it all at once
- **Bottom safe margin:** leave **extra empty space at the bottom** of every slide — the video editor's control panel pops up there and can cover content

---

## 🎬 Slide 0 — Title

**Title:** How AI Agents Get Smarter — Without Retraining the Model

**On-slide:**
- Model = **frozen** 🧊
- Yet agents **get smarter**
- How? The **system** grows

**Visual:** A frozen brain (🧊) center, surrounded by growing "filing cabinets" of knowledge feeding into it.

**Animate:** Frozen brain appears locked in the center, then filing cabinets pop in around it one by one and arrows feed into the brain.

**Speaker notes:**
"We often say an AI agent 'learns from past data and becomes an expert over time.' But if the model is trained once and frozen, how does that work? In this short tutorial we'll clear up that paradox: the model itself doesn't get smarter — the system you build around it does. We'll see how, and what tools you can use to build it."

---

## 🧊 Slide 1 — The Paradox: Frozen Model, "Learning" Agent

**Title:** The Model Has Amnesia — But the Agent Remembers

**On-slide:**
- **Frozen · stateless**
- Genius **reasoner**, no memory
- Knowledge lives **outside**
- …and gets **fed in**

**Visual:**
```
   ticket ─▶ [ FROZEN LLM (reasons, forgets) ] ─▶ answer
                        ▲
              knowledge injected from OUTSIDE
```

**Animate:** Ticket flows into the frozen LLM and an answer comes out; then the "knowledge injected from OUTSIDE" arrow drops in from above and glows.

**Speaker notes:**
"Here's the paradox. A trained model has frozen weights and is stateless — it reasons brilliantly but remembers nothing between calls. Think of it as a genius contractor with amnesia, re-hired fresh for every single task. So how does an agent become an expert? The trick is that the knowledge doesn't live in the model — it lives outside, in your system, and gets fed into the model's context every time."

---

## 🧠 Slide 2 — How Agents "Learn" Without Retraining

**Title:** The System Gets Smarter, Not the Weights

**On-slide:**
- **RAG** ⭐ retrieve past cases
- **Knowledge base** grows
- **Feedback** ranks results
- **Tools** pull live data

**Visual:**
```
new task ─▶ retrieve similar past cases ─▶ inject into prompt ─▶ [ LLM ] ─▶ answer
   ▲                                                                   │
   └──────────── save resolution back to knowledge base ◀─────────┘
```

**Animate:** New task retrieves past cases (arrow in), the LLM answers, then the "save resolution back" loop arrow animates and the knowledge base visibly grows.

**Speaker notes:**
"So how does the agent 'learn'? Mainly through retrieval, or RAG. Every past case — resolved tickets, runbooks, resolutions — goes into a knowledge base, usually a vector database. When a new task arrives, the agent finds the most similar past cases and injects them into the prompt. The model 'recalls' them, not because it learned them, but because your system retrieved them. Add a feedback loop so engineers can mark answers good or bad, tools that pull live data at run time, and better prompts over time — and the whole system compounds. More data means better retrieval means smarter answers. That's the 'expert over time' effect, with zero retraining."

---

## 🧩 Slide 3 — The Model Team (Who Retrieves & Who Summarizes?)

**Title:** It's Not One LLM — It's a Team of Models

**On-slide:**
- **Embeddings** retrieve (cheap)
- **Reranker** sharpens
- **Small LLM** summarizes
- **BIG LLM** reasons

**Visual:**
```
text ─(embedding model, cheap)─▶ vectors ─▶ vector DB ─(similarity)─▶ top hits
                                                          │ (reranker, cheap)
                                                          ▼
                        (small LLM) summarize/compact ─▶ [ BIG LLM ] final answer
```

**Animate:** Text turns into vectors → nearest hits light up in the DB → reranker trims them → small LLM compacts → the **BIG LLM** box glows last for the final answer.

**Speaker notes:**
"A natural question: to retrieve the right data and summarize context, don't we need an LLM — and who does it? The key insight is that a good agent isn't one big LLM; it's a *team* of models, each with a job. Retrieval itself mostly doesn't use the big chat model — it uses a cheap embedding model that turns text into vectors, and then plain math finds the nearest ones. An optional reranker, another small model, sharpens the top results. Summarization *does* need an LLM, but you route it to a smaller, cheaper one — and ideally you summarize once, at write-time, when a ticket resolves, so it's reused for free every time after. Your expensive top-tier model is saved for the final reasoning step only. This is model routing: cheap specialized models for the grunt work, the big model for the hard thinking. And note — none of this needs fine-tuning; though if you ever do fine-tune, sharpening the *embedding model* for your domain often helps retrieval more than touching the chat model."

---

## 🎫 Slide 4 — Example: Sev2 Ticket Investigator

**Title:** A Real Agent — Getting Expert at Incidents

**On-slide:**
- Past incidents → **vector store**
- New Sev2 → **retrieve** + runbooks
- Pull **live logs / metrics**
- Reason → **fix** → **save** 🔁

**Visual:** Flow: `Sev2 ticket ▶ retrieve past incidents ▶ pull live logs ▶ LLM reasons ▶ fix ▶ save outcome ▶ (loop)`.

**Animate:** The flow lights up step by step left→right; the final "save outcome" arrow loops back and the vector store grows a notch.

**Speaker notes:**
"Let's ground it in a real example — a Sev2 ticket investigator. Every resolved incident, with its root cause and fix, is stored in a vector database. When a new Sev2 fires, the agent retrieves similar past incidents and the relevant runbooks, then calls tools to pull the current logs, metrics, and recent deploys. The frozen model reasons over all of that and suggests a root cause and fix. Then the outcome is saved back — so next time a similar incident appears, the agent is even sharper. The model never changed; the knowledge around it grew."

---

## 🛠️ Slide 5 — Do You Build It From Scratch? (Agent Frameworks)

**Title:** Use a Framework for Plumbing — Don't Reinvent It

**On-slide:** (pick one; all handle the agent loop, tools, memory, retrieval)

| Framework | Good for |
|---|---|
| **LangGraph / LangChain** | Stateful, controllable agents |
| **LlamaIndex** | RAG-heavy apps |
| **OpenAI Agents SDK** | Lightweight tool-calling |
| **Semantic Kernel** | Enterprise / .NET & Python |
| **CrewAI / AutoGen** | Multi-agent teams |

**Visual:** A toolbox labeled "plumbing: tool-calling, memory, retrieval, tracing".

**Animate:** Rows reveal one at a time; a toolbox icon fills up with **tool-calling · memory · retrieval · tracing** as each row appears.

**Speaker notes:**
"Do you build all this from scratch? No. There are solid agent frameworks that handle the plumbing — the tool-calling loop, memory interfaces, retrieval connectors, retries, and tracing. LangGraph and the OpenAI Agents SDK are strong current defaults; LlamaIndex is great if you're retrieval-heavy; Semantic Kernel suits enterprise stacks; CrewAI and AutoGen focus on multiple agents working together. Pick one so you're not reinventing the loop."

---

## 🏆 Slide 6 — Will One Team's Agent Beat Another's?

**Title:** Same Framework, Same Model — Different Quality

**On-slide:**
- Framework = **plumbing**, not quality
- Quality = **your engineering**
- **Data · retrieval · context**
- **Evaluation loop** = the moat

**Visual:** Two teams, same "framework + model" box, but different outputs — one 💸 noisy, one ✅ sharp.

**Animate:** Both teams start from an identical box; their outputs diverge — one turns 💸 noisy, the other ✅ sharp — and "evaluation loop" highlights on the winning side.

**Speaker notes:**
"So if two teams use the same framework and the same model, will one be better? Yes — often dramatically. The framework only gives you plumbing; it can't make the hard decisions that define quality. Those stay with you: the quality of your data and retrieval, whether you curate the *right* context instead of dumping everything, how reliable your tools are, how good your prompts and guardrails are, and — most importantly — whether you have an evaluation loop to measure answer quality and improve it. That evaluation discipline is the real moat. The framework is table stakes; your engineering around it is the differentiator."

---

## 🎯 Slide 7 — Key Takeaways

**Title:** Key Takeaways

**On-slide:**
- Model stays **frozen** 🧊
- System **feeds context**
- Knowledge **compounds**
- It's a **team of models**
- **Quality = your engineering**

**Visual:** Five-point checklist that ticks in one by one.

**Animate:** Each takeaway ticks in one at a time with a ✅.

**Speaker notes:**
"Let's recap. The model is frozen and doesn't learn while serving. Agents become experts because the system around them feeds better context every time — through retrieval, a growing knowledge base, feedback, and live tools. That knowledge lives outside the model and compounds. And it's not one model doing everything — it's a team: cheap embedding models handle retrieval, a small LLM summarizes, and the big model is saved for the final reasoning. Use an agent framework so you don't rebuild the plumbing. But remember: quality comes from your engineering — clean data, strong retrieval, curated context, reliable tools, and an evaluation loop. That's what makes one agent an expert and another mediocre."
