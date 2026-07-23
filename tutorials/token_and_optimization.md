# Tokens, Tokenizers, Context Windows & Token Optimization

> **Tutorial format:** Each section below = **one slide** (or a small group of slides).
> Each slide has: a **Title**, **On-slide bullets** (short, for the PPT), a **Visual** suggestion,
> and **Speaker notes** (what you say in the video). Keep on-slide text short; say the rest.

---

## 🎨 Slide Design Guidelines (apply to every slide)

- **Background:** full **black** (`#000000`)
- **Title:** **bold**, **dark-orange** (`#CC5500` / `#D2691E`)
- **Body text:** **all bold**, mostly **white** (`#FFFFFF`)
- **Accents (optional):** use the same dark-orange sparingly for emphasis/keywords
- **Font:** clean sans-serif (e.g. Inter, Montserrat, Segoe UI); large sizes for readability
- **Contrast:** keep high — white/orange on black; avoid low-contrast greys
- **Bottom safe margin:** leave **extra empty space at the bottom** of every slide — the video editor's control panel pops up there and can cover content

---

## 🎬 Slide 0 — Title

**Title:** How AI Reads Text — Tokens, Tokenizers, Context Windows & Optimization

**On-slide:**
- The hidden "unit" behind every LLM
- Why it controls **cost, speed, and memory**
- A beginner-friendly deep dive

**Visual:** Big title, a subtle animation of text `"Hello world"` turning into numbers `[9906, 1917]`.

**Speaker notes:**
"Every time you chat with an AI like GPT, Claude, or Llama, your words are secretly converted into numbers before the model sees them. In this video we'll uncover that hidden layer — tokens and tokenizers — then see how it defines the context window, and finally learn how engineers optimize tokens to save money and speed things up."

---

## 🧭 Slide 1 — What We'll Cover (Roadmap)

**Title:** The Journey

**On-slide:**
1. What is a **token**?
2. What is a **tokenizer**?
3. Do all models tokenize the same? (GPT vs Claude vs Llama)
4. What is the **context window**?
5. **Token optimization** — do more with less
6. Live demo + key takeaways

**Visual:** A horizontal path/roadmap with 6 numbered stops.

**Speaker notes:**
"Here's our roadmap. We'll build each concept on top of the last, so by the end you'll understand not just *what* tokens are, but *why they matter* for anyone building with AI."

---

## 🧩 Slide 2 — The Core Idea: Models Only Understand Numbers

**Title:** Models Don't Read Words — They Read Numbers

**On-slide:**
- Text ➜ **numbers** ➜ model ➜ **numbers** ➜ text
- The model does **math**, not spelling
- Something must translate. That's the tokenizer.

**Visual:**
```
"Hello world"  ──encode──▶  [9906, 1917]  ──▶  🧠 MODEL
     text                     numbers
```

**Speaker notes:**
"Here's the one idea that everything rests on: language models are giant math machines. They can't read letters or words directly — they only work with numbers. So before any text reaches the model, it has to be turned into numbers. And the model's reply comes back as numbers that get turned back into text for you to read."

---

## 🔤 Slide 3 — What Is a Token?

**Title:** A Token = a Chunk of Text

**On-slide:**
- The **basic unit** a model reads/writes
- Not always a whole word — often a **piece** of one
- Rough rule (English): **1 token ≈ 4 characters ≈ ¾ of a word**

**Visual:** Show one sentence split into colored blocks:
```
"Tokenization is powerful"
 ┌────────┬──────────┬────┬──────────┐
 │ Token  │ ization  │ is │ powerful │
 └────────┴──────────┴────┴──────────┘
     4 tokens
```

**Speaker notes:**
"A token is simply a chunk of text — the basic unit the model reads and writes. Notice it's not always a full word. Common words like 'is' are one token, but a longer word like 'tokenization' gets split into 'token' + 'ization'. As a rough rule of thumb for English, one token is about four characters, or roughly three-quarters of a word."

---

## 🧱 Slide 4 — Why Split Into Pieces? (Sub-words)

**Title:** Why Break Words Into Pieces?

**On-slide:**
- Common words ➜ **1 token** (compact)
- Rare/new words ➜ **several sub-word tokens**
- Result: model can represent **any** word — even typos & new slang
- No "unknown word" problem

**Visual:**
```
"cat"              ➜ [cat]                     (1 token)
"antidisestablish" ➜ [anti][dis][establish]   (sub-words)
"AImazing"         ➜ [AI][maz][ing]           (handles new words!)
```

**Speaker notes:**
"Why bother splitting words? Because it's the perfect balance. Common words stay as a single compact token. Rare or brand-new words get broken into smaller sub-word pieces. This means the model can represent literally any text you throw at it — including typos, new slang, or made-up words — by assembling smaller building blocks. It never gets stuck on an 'unknown word'."

---

## ⚙️ Slide 5 — What Is a Tokenizer?

**Title:** The Tokenizer — Text ⇄ Numbers Translator

**On-slide:**
- The tool that **encodes** text ➜ tokens ➜ IDs (and **decodes** back)
- Two jobs: **1) split** the text, then **2) look up** each piece
- **Vocabulary** = a dictionary/hashmap: `text piece ⇄ ID` ✅ instant lookup
- **Tokenizer** = the **algorithm** that decides *how to split* — the smart part

**Visual:**
```
        ┌──────────────── TOKENIZER ────────────────┐
text ─▶ │  1) SPLIT (algorithm)   2) LOOKUP (map)    │ ─▶ [IDs]
        └────────────────────────────────────────────┘

   VOCABULARY (hashmap)
    "cat"     -> 2543      "token"   -> 9906
    "ization" -> 1634      " world"  -> 1917
```

**Speaker notes:**
"The tokenizer is the translator that sits between you and the model. It does two things: first it *splits* your text into token pieces, then it *looks up* each piece to get its number, called a token ID. It also runs in reverse to turn the model's number output back into text.

A common question here: isn't a token just a hashmap from text to a number? Half right. The *vocabulary* — the list of known pieces and their IDs — is exactly a hashmap, and looking up a piece is instant. But the *tokenizer* is more than that: it's the algorithm that decides *how* to chop the text into those pieces in the first place. You can't just look up the whole word 'tokenization' — the tokenizer has to figure out the best split first, then look up each piece."

---

## 🔬 Slide 6 — How Splitting Works (BPE in 10 seconds)

**Title:** How Splitting Works — Byte Pair Encoding (BPE)

**On-slide:**
- Start from single characters
- **Greedily merge** the most frequent pairs
- Repeat using rules learned during training
- Stop at the largest valid tokens

**Visual:**
```
t o k e n i z a t i o n
   ▼ merge frequent pairs ▼
to  ke  n  iz  a  ti  o  n
   ▼
token   ization
```

**Speaker notes:**
"Here's the intuition behind the most common method, called Byte Pair Encoding, or BPE. It starts with individual characters and repeatedly merges the pairs that appear most often together — following merge rules it learned from huge amounts of text during training. It keeps merging until it reaches the biggest valid tokens. That's how 'tokenization' ends up as 'token' plus 'ization'."

---

## 🌐 Slide 7 — Do All Models Tokenize the Same?

**Title:** GPT vs Claude vs Llama — Same Text, Different Tokens

**On-slide:**
- **No** — each model family has its **own tokenizer & vocabulary**
- Same sentence ➜ slightly **different token counts**
- The ~4-chars-per-token rule is *approximate*, not exact

**Visual:** Table.

| Model family | Tokenizer | Vocab size (approx) |
|---|---|---|
| OpenAI GPT | BPE (`o200k_base`) | ~200K |
| Anthropic Claude | Proprietary BPE | — |
| Meta Llama 3 | BPE / SentencePiece | ~128K |
| Google Gemini | SentencePiece | — |

**Speaker notes:**
"Do all models tokenize the same way? No. Each model family trains its own tokenizer with its own vocabulary. So the exact same sentence might be 20 tokens for GPT, 22 for Claude, and 24 for Llama. They're all roughly similar — which is why the four-characters-per-token rule works as a ballpark — but the exact counts differ. That matters when you compare cost or context limits across models."

---

## 🔁 Slide 8 — What About Version Bumps? (Opus 4.7 vs 4.8)

**Title:** Minor Updates Keep the Tokenizer; Major Ones May Change It

**On-slide:**
- **Minor** version (e.g. Opus 4.7 ➜ 4.8): usually **same** tokenizer
- **Major** generation (e.g. Llama 2 ➜ 3, GPT-3 ➜ 4): tokenizer **can change**
- Changing the vocab requires retraining & breaks compatibility

**Visual:** Timeline: `4.7 ──(same tokenizer)── 4.8` vs `Llama 2 ──(new tokenizer)── Llama 3`.

**Speaker notes:**
"What about small updates, like Claude Opus 4.7 to 4.8? Those almost always keep the same tokenizer, so token counts don't change. Tokenizers usually only change across *major* generations — like Llama 2 to Llama 3 — because swapping the vocabulary means retraining the model and breaks compatibility with the old encoding."

---

## 🪟 Slide 9 — The Context Window

**Title:** Context Window — The Model's Working Memory

**On-slide:**
- Max tokens a model can "see" **at once**
- Covers **input + output** together
- Examples: 128K, 200K, 1M tokens
- Exceed it ➜ older text gets **dropped or truncated**

**Visual:**
```
┌──────────── CONTEXT WINDOW (e.g. 200K tokens) ────────────┐
│ system prompt │ chat history │ your question │ answer      │
└────────────────────────────────────────────────────────────┘
        overflow ⟶ old text falls off the edge
```

**Speaker notes:**
"Now the context window. Think of it as the model's short-term working memory — the maximum number of tokens it can hold in view at one time. Crucially, this budget covers *both* your input and the model's output. If a conversation grows past the limit, the oldest content has to be dropped or truncated — the model literally can't see it anymore. This is why long chats sometimes 'forget' what you said at the start."

---

## 💸 Slide 10 — Why Tokens Matter (Cost, Speed, Memory)

**Title:** Why You Should Care About Tokens

**On-slide:**
- 💰 **Cost** — you're billed **per token** (input & output priced separately)
- ⚡ **Speed** — more tokens = slower responses
- 🧠 **Memory** — the context window is finite
- 🎯 **Quality** — bloated context ➜ "lost in the middle" errors

**Visual:** Four icon cards: money, lightning, brain, target.

**Speaker notes:**
"So why care? Four reasons. Cost: APIs charge per token, and input and output have separate prices. Speed: more tokens means more to process, so slower replies. Memory: the context window is finite. And quality: when you cram too much in, models can lose track of important details buried in the middle. All four push us toward one goal — using fewer, better tokens."

---

## 🧾 Slide 11 — How You're Billed

**Title:** Do All AI Companies Charge Per Token?

**On-slide:**
- **APIs (developers):** almost always **per token**
  - Input & output priced separately — **output costs more** (often 3–5×)
  - **Cached** input is heavily discounted ➜ optimization pays off
- **Consumer apps:** flat **subscription** (tokens hidden) — e.g. ChatGPT Plus
- **Open-weight models** (Llama, Mistral): **free to run** — you pay only **hardware/cloud**

**Visual:** Three cards — 🔌 *API = per token* · 📅 *App = subscription* · 💻 *Self-host = compute cost*.

```
cost = (input_tokens × input_rate) + (output_tokens × output_rate)
```

**Speaker notes:**
"A natural question: does everyone charge per token? For developer APIs — OpenAI, Anthropic, Google, Mistral — yes, per-token billing is the near-universal standard. Input and output are priced separately, and output usually costs three to five times more because generating is harder than reading. Reused cached input is billed at a big discount, which is exactly why optimization saves money. But there are exceptions. Consumer apps like ChatGPT Plus hide tokens behind a flat monthly subscription. And open-weight models like Llama or Mistral are free to download — there's no per-token charge at all; you just pay for the hardware or cloud you run them on. So: APIs bill per token, subscriptions bundle it, and self-hosting turns it into a compute cost."

---

## ❄️ Slide 12 — The Input Snowball (One Session, Many Turns)

**Title:** Every Turn Re-Sends the Whole Conversation

**On-slide:**
- Models are **stateless** — they remember nothing between calls
- So each turn re-sends the **entire history** as input
- Your new question **+ all past questions AND past answers** = this turn's input
- Past **output** becomes next turn's **input** ➜ input **snowballs**

**Visual:** A growing stack; table below.

```
Turn │ INPUT sent (grows!)              │ OUTPUT
─────┼──────────────────────────────────┼───────
 Q1  │ system + Q1                      │  A1
 Q2  │ system + Q1 + A1 + Q2            │  A2
 Q3  │ system + Q1 + A1 + A2 + Q3       │  A3
```

**Speaker notes:**
"Here's something that surprises almost everyone. In a chat or agent session, is only your latest question the input? No — the whole conversation is. The model is stateless; it remembers nothing between calls. So to stay coherent, the app re-sends everything every turn: the system prompt, all your previous questions, AND all the model's previous answers, plus your new message. That means past output becomes future input. Look at the table — by question three you're re-paying for Q1, A1, and A2 all over again. The input snowballs, so later turns cost far more than the first. This is the single biggest reason long chats get expensive — and the reason the optimization tricks we'll see next, especially prompt caching and summarization, matter so much."

---

## 🧠 Slide 13 — Why Not Just Store It Server-Side? (Compute ≠ Transmission)

**Title:** "Why Doesn't the Provider Store My Session So Cost Is Equal?"

**On-slide:**
- **The real reason:** providers can't auto-trim — **only the client knows what's relevant** (+ privacy/control)
  - Context is often **large** (whole codebase), **live-changing** (files edited between turns), or **external** (DBs, APIs, private docs) → must be assembled **fresh, client-side**
- Cost isn't from **sending** history — it's from the model **re-reading** it (compute)
- Stored in VS Code or on the provider's server → model **still reprocesses every token**
- Server-side sessions **exist** (OpenAI Responses/Assistants, threads) — but you're **still billed** for those tokens
- The only real compute saver = **prompt/KV cache** (reuse computed prefix)

**Visual:** Two paths into the same "GPT-4 reprocesses all tokens" box — one labeled *history in client*, one *history on server* — both same cost. A separate ✅ branch: *cached prefix → skip compute*.

**Speaker notes:**
"Great question people always ask: why can't the model holder just store my session on their side, so every caller pays the same and nobody re-implements this? There are actually two independent reasons. The first is correctness and access: only the client knows what's relevant *right now*. Think about a coding assistant — the context might be a whole codebase that's millions of tokens, far too big to store and send in full; it's live-changing, because you edit files between turns, so a server-side snapshot would go stale and give wrong answers; and it often comes from external sources like databases, internal APIs, or private docs that live behind the client's own authentication. So the relevant context has to be assembled fresh, client-side, every turn. The second reason is cost: even if the provider *did* store it, the cost doesn't come from *transmitting* the history — it comes from the model *re-reading* it. Every turn, the model pushes every token through its attention layers before it can reply. Whether that text lives in VS Code or on OpenAI's servers, the model still reprocesses all of it — same tokens, same compute, same cost. In fact server-side sessions already exist — OpenAI's Responses and Assistants APIs keep the thread for you — but you're still billed for every history token, because it's reprocessed. The one thing that genuinely cuts compute is the prompt or KV cache: reusing the already-computed attention state for an unchanged prefix. So between dynamic, private, client-owned data on one side and the reprocessing cost on the other, orchestration stays the client's job."

---

## 🚀 Slide 14 — Token Optimization: The Big Picture

**Title:** Token Optimization — Do More With Less

**On-slide:**
- Agents are **token-hungry**: they loop and re-send context every step
- Optimization = keep only what's **relevant**
- Saves **money**, cuts **latency**, protects **quality**

**Visual:** Before/after: a bloated prompt shrinking into a lean one.

**Speaker notes:**
"This is where token optimization comes in. AI agents are especially token-hungry because they work in a loop — every step re-sends the system prompt, the whole history, tool descriptions, and tool outputs back to the model. Without care, that grows fast. Optimization is the art of keeping only what's relevant, so you spend less, respond faster, and keep quality high."

---

## 🧰 Slide 15 — Optimization Techniques (Part 1: Context)

**Title:** Technique Group 1 — Manage the Context

**On-slide:**
- **Trimming / windowing** — drop old, irrelevant messages
- **Summarization / compaction** — replace long history with a short summary
- **Retrieval (RAG)** — fetch only the relevant chunks, not whole documents

**Visual:** Three mini-diagrams: scissors (trim), compress arrows (summarize), magnifying glass over a doc (retrieve).

**Speaker notes:**
"Let's get practical. The first group of techniques is about managing the context itself. Trimming means dropping old messages that no longer matter. Summarization, or compaction, replaces a long history with a compact summary — many agents do this automatically when they near the limit. And retrieval, or RAG, means instead of pasting an entire document, you fetch just the few relevant paragraphs you actually need."

---

## 🧰 Slide 16 — Optimization Techniques (Part 2: Prompts & Tools)

**Title:** Technique Group 2 — Lean Prompts & Tools

**On-slide:**
- **Concise prompts** — trim filler in system/instructions
- **Load only needed tools** — don't send every tool definition
- **Compact tool outputs** — filter/paginate/summarize before feeding back
- **Cap verbose logs** before returning to the model

**Visual:** A toolbox where only the needed tools light up.

**Speaker notes:**
"The second group is about lean prompts and tools. Keep system prompts concise. Only load the tool definitions the agent actually needs right now, instead of sending all of them every turn. And when a tool returns a giant blob — like logs or a big API response — filter, paginate, or summarize it before feeding it back, instead of dumping raw data into the context."

---

## 🧰 Slide 17 — Optimization Techniques (Part 3: Caching & Routing)

**Title:** Technique Group 3 — Caching & Smart Routing

**On-slide:**
- **Prompt caching** — reuse the encoded stable prefix; don't re-pay for it
- **Model routing** — send simple subtasks to smaller/cheaper models
- **Subagents** — do heavy exploration in isolation; return only a summary

**Visual:** A cache icon + a signpost routing tasks to "small model" vs "big model".

**Speaker notes:**
"The third group is caching and smart routing. Prompt caching lets you reuse the encoded system prompt or any stable prefix across calls, so you're not re-paying for the same tokens every time. Model routing sends simple subtasks to smaller, cheaper models and saves the big model for hard problems. And subagents let heavy exploration happen in a separate context, returning only a short summary to the main conversation — keeping the main window clean."

---

## 🧪 Slide 18 — Live Demo: Count Tokens in Python

**Title:** Demo — See Tokenization in Action

**On-slide:**
- Use OpenAI's `tiktoken` library
- Encode ➜ see IDs ➜ decode back
- Try English vs code vs emoji

**Visual:** Code block + expected output.

```python
import tiktoken

enc = tiktoken.get_encoding("o200k_base")

text = "Hello, tokenizer!"
ids = enc.encode(text)

print(ids)            # [13225, 11, 90985, 0]  (example IDs)
print(len(ids))       # 4  -> token count
print(enc.decode(ids))  # "Hello, tokenizer!"
```

**Speaker notes:**
"Let's make it real. Using OpenAI's open-source `tiktoken` library, we encode a sentence into token IDs, count them, and decode them back to text. On camera, try encoding plain English, then a line of code, then some emoji — you'll see code and emoji use far more tokens per character than English. That's a great visual for why tokenization efficiency matters."

---

## 🏆 Slide 19 — Why the Client Wins (Orchestration Is the Moat)

**Title:** Same Model, Different Bill — The Client Decides

**On-slide:**
- The model is **stateless** & priced **per token** by the provider — that's **fixed**
- **How many tokens get sent** is decided by the **client** (VS Code, Cursor, Claude Code, ChatGPT app…)
- Same task on the same GPT-4 can cost **many× more or less** — purely from orchestration
- The moat = context management + caching + routing + retrieval
- ⚠️ Goal is **lean *and* relevant** — over-trimming hurts quality

**Visual:** Same "GPT-4" box fed by two clients — a bloated prompt (💸) vs a lean one (✅).

| Client | Optimization edge |
|---|---|
| VS Code Copilot / Cursor | Send only relevant code chunks, smart context selection |
| Claude Code | Auto-compaction, cache-friendly prefixes, subagents |
| ChatGPT web/mobile | Flat fee — *provider* absorbs the cost |
| Raw API (your code) | You own 100% of the token bill |

**Speaker notes:**
"Here's the big takeaway that ties everything together. The GPT model is basically a commodity — it's stateless and priced per token, and that price is fixed by the provider. But *how many tokens get sent* is decided entirely by the client and its orchestration layer. That means two tools calling the exact same GPT-4 can differ many-fold in cost for the same task, purely based on how well they manage context — trimming, summarizing, prompt caching, routing, and retrieval. So the real competitive moat for tools like Cursor, Claude Code, and Copilot isn't the model — it's the orchestration around it. One caveat: the goal is lean *and* relevant. Over-aggressive trimming saves tokens but drops context the model needed, hurting quality. Whoever keeps context lean and relevant delivers the same quality for less money and lower latency. That's where the game is won."

---

## 🎯 Slide 20 — Key Takeaways

**Title:** Key Takeaways

**On-slide:**
- **Token** = the chunk of text a model reads/writes (~¾ word)
- **Tokenizer** = algorithm that *splits* text + hashmap that *looks up* IDs
- Token counts **vary by model family**, stable across minor versions
- **Context window** = finite input+output memory budget
- Sessions **snowball** — every turn re-sends the whole history as input
- **Optimization** lives in the **client** ➜ lean + relevant = cheaper, faster, sharper

**Visual:** Six-point checklist that ticks in one by one.

**Speaker notes:**
"Let's recap. A token is a chunk of text, about three-quarters of a word. A tokenizer is an algorithm that splits text plus a hashmap that looks up IDs. Token counts vary between model families but stay stable across minor version updates. The context window is your finite memory budget for input and output combined. In a session, input snowballs because every turn re-sends the whole history. And optimization lives in the client — keeping context lean *and* relevant is what makes your AI cheaper, faster, and more accurate."

---

## 👋 Slide 21 — Outro

**Title:** Thanks for Watching!

**On-slide:**
- Next up: *(your next topic / playlist link)*
- Like • Subscribe • Comment your questions
- Try the `tiktoken` demo yourself!

**Visual:** End card with subscribe button and next-video thumbnail.

**Speaker notes:**
"That's tokens, tokenizers, context windows, and optimization in one video. If this helped the concept click, hit like and subscribe, and drop your questions in the comments. Try the tiktoken demo yourself and tell me the weirdest thing you tokenized. See you in the next one!"

---

## 📎 Appendix — Quick Glossary (for description box / handout)

| Term | One-line definition |
|---|---|
| **Token** | The basic chunk of text a model processes (~4 chars / ~¾ word in English). |
| **Token ID** | The integer a token maps to in the vocabulary. |
| **Vocabulary** | The fixed hashmap of all known token pieces ⇄ IDs. |
| **Tokenizer** | Algorithm that splits text into tokens, then looks up their IDs (and reverses it). |
| **BPE** | Byte Pair Encoding — builds tokens by greedily merging frequent character pairs. |
| **Context window** | Max tokens a model can process at once (input + output combined). |
| **Prompt caching** | Reusing an encoded stable prefix so you don't re-pay for the same tokens. |
| **RAG** | Retrieval-Augmented Generation — fetch only relevant chunks into the prompt. |
| **Compaction** | Replacing long history with a short summary to save tokens. |
