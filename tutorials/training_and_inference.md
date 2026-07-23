# Training vs Inference — The Two Halves of AI Engineering

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

**Title:** Training vs Inference — The Two Halves of AI Engineering

**On-slide:**
- **Two phases** 🧠
- 📚 **TRAINING** — teach
- 💬 **INFERENCE** — use

**Visual:** Split screen — 📚 brain "studying" (left) · 💬 brain "answering" (right).

**Animate:** Left half fades in with **TRAINING**, right half slides in with **INFERENCE**, then the title drops in last.

**Speaker notes:**
"Behind every AI product there are two completely different jobs. First, someone has to *teach* the model — that's training. Then, millions of times a day, someone *uses* that trained model to get answers — that's inference. They look similar from the outside, but under the hood they have opposite goals, run on different budgets, and even use hardware differently. In this video we'll unpack both halves and see how they fit together."

---

## 🧩 Slide 1 — The Big Picture: Two Phases

**Title:** AI Has Two Modes — Learn, Then Do

**On-slide:**
- **Train** → learn → **weights**
- **Infer** → use weights
- Train rarely · infer always

**Visual:**
```
   DATA ──▶ [ TRAINING ] ──▶ MODEL WEIGHTS ──▶ [ INFERENCE ] ──▶ ANSWERS
             (learn)          (the "brain")      (use)
```

**Animate:** Pipeline builds left→right, one box at a time; **MODEL WEIGHTS** pulses as the handoff between the two phases.

**Speaker notes:**
"Here's the one mental model to hold onto. Training is where the model *learns* — you feed it huge amounts of data, and it produces a set of numbers called weights, which are essentially the model's brain. Inference is where the model *uses* that brain to respond to brand-new inputs it has never seen. The key rhythm: you train relatively rarely — maybe once, then occasionally to update — but you run inference constantly, on every single user request."

---

## 🎓 Slide 2 — What Is Training?

**Title:** Training — Teaching the Model

**On-slide:**
- Feed **data + answers**
- Guess → check → **fix**
- Repeat billions ×
- Out = **weights**

**Visual:**
```
      ┌─────────── TRAINING LOOP ───────────┐
data ▶│ predict ▶ measure error (loss) ▶ adjust weights │▶ better model
      └──────────────────────────────┘
                (repeat millions/billions of times)
```

**Animate:** The loop arrows cycle continuously; a counter ticks **1 → millions**, and "better model" brightens each pass.

**Speaker notes:**
"Let's zoom into training. You show the model an example, it makes a prediction, and you measure how wrong that prediction was — that error is called the loss. Then an algorithm nudges the model's weights slightly to reduce that error. Do this over and over, billions of times, across enormous datasets, and the model gradually encodes the patterns of language, images, or whatever you're teaching it. The end product is a trained model — literally a big file of numbers, the weights."

---

## ⚙️ Slide 3 — How Training Works (Gradient Descent, Simply)

**Title:** The Learning Engine — Loss & Gradient Descent

**On-slide:**
- **Loss** = how wrong
- Roll **downhill** ⬇️
- Nudge the weights
- Passes = **epochs**

**Visual:** A ball rolling down a curved valley toward the lowest point ("minimize loss").

**Animate:** Ball drops onto the curve and rolls to the bottom, bouncing to a stop; a **"min loss"** flag pops up.

**Speaker notes:**
"How does the model actually improve? Picture a hilly landscape where height is the loss — how wrong the model is. Learning means rolling downhill toward the lowest point. Gradient descent is the method: it computes the slope and steps the weights in the direction that reduces error. Backpropagation is the bookkeeping that tells us exactly which of the millions of weights to adjust and by how much. Each full pass over the training data is called an epoch, and we do many of them."

---

## 🧮 Slide 4 — What Are Weights? (Embeddings & Fixed Shapes)

**Title:** Weights Are Numbers — Stored as Vectors in Fixed-Size Tables

**On-slide:**
- Weights = **numbers**
- 1 token = **1 vector**
- Shape **V × d** (fixed)
- Related = **nearby** 📍

**Visual:**
```
            d columns (hidden size, fixed)  →
         ┌───────────────────────────────┐
  cat    │ 0.95  0.10 -0.44 ...  (d nums) │  ← near MANY tokens
  token  │ 0.12 -0.87  0.33 ...           │
  ...    │  ...                          │
  (V     │                               │
  rows)  │                               │
         └───────────────────────────────┘
     rows = V (vocab)   ×   cols = d   =   V×d weights
```

**Animate:** Table fills **row by row**; then a small "space" inset appears where two related tokens (e.g. *cat* / *dog*) light up **close together**.

**Speaker notes:**
"So what exactly are these weights? They're just numbers — but organized into fixed-size tables. The most intuitive one is the embedding table: it has exactly one row, one vector, per token in the vocabulary. Its shape is V by d — V is the vocabulary size, and d is the hidden size, the length of every vector. Both are fixed before training and never change; training only fills in the values. Here's the beautiful part: every token's vector is the *same* length, d — a few thousand numbers — even though 'cat' might relate to thousands of tokens and a rare word to only a few. How? Because a token isn't a list of relationships; it's a *point* in a d-dimensional space, and 'related' just means 'nearby.' Just like a city is two GPS numbers whether it has thousands of neighbors or almost none — the count of neighbors comes from *where* it sits, not from the size of its coordinates. Meaning is spread across all d numbers together, which is why a fixed-length vector can encode an essentially unlimited web of relationships."

---

## 💬 Slide 5 — What Is Inference?

**Title:** Inference — Using the Trained Model

**On-slide:**
- Weights **frozen** 🧊
- In → compute → **out**
- Every request
- **Fast · cheap · right**

**Visual:**
```
   new input ──▶ [ FROZEN MODEL (weights) ] ──▶ prediction / answer
                        no weight changes
```

**Animate:** Input arrow flies in, the model pulses once ("thinking"), the answer pops out; a 🧊 icon stays locked on the weights.

**Speaker notes:**
"Inference is the everyday job. The weights are frozen now — the model isn't learning anymore, it's just *using* what it learned. You give it a new input, it runs the math forward through those fixed weights, and out comes a prediction or an answer. This is exactly what happens every time you send a chat message or call an API. Here the goals flip: instead of accuracy over a giant dataset, you care about serving each request fast, cheaply, and correctly."

---

## 🎯 Slide 6 — How Inference Picks the Next Token

**Title:** Picking the Next Token — Transform First, Then Compare

**On-slide:**
- **Embed** → vectors
- **Transform** (≈99%)
- **Compare** (dot-product)
- **Softmax** → **pick**

**Visual:**
```
tokens ▶ embed ▶ [ attention + feed-forward layers ] ▶ final vector (d)
                                                          │ dot-product with V output vectors
                                                          ▼
                                      V logits ▶ softmax ▶ pick next token
```

**Animate:** Each stage lights up in sequence; the **Transform** block glows biggest ("99% of the work"), then the winning token **bounces out** at the end.

**Speaker notes:**
"So how does inference actually choose the next word? A common guess is 'it measures distance in the vector space' — that's half right. Here's the real flow. First, each input token is turned into its vector. Then the bulk of the work — about ninety-nine percent of the compute — runs those vectors through all the attention and feed-forward layers, which use context to build a single final vector that summarizes 'what should come next.' Only *then* comes the similarity step: the model takes a dot product between that final vector and every token's output vector, giving one score per vocabulary token. Higher score means better aligned in the space. Softmax turns those scores into probabilities, and the model picks one. So distance-like similarity decides the *final choice*, but the heavy lifting is the transformation that *produces* the vector being compared."

---

## 🧊 Slide 7 — Does the Model Learn While Serving? (No — Frozen & Stateless)

**Title:** Inference Is Read-Only — The Model Doesn't Learn From You

**On-slide:**
- No learning while serving
- **Stateless** — forgets you
- "Memory" = **re-sent text**
- Real learning = **retrain**

**Visual:**
```
request ─▶ [ FROZEN weights ] ─▶ response ─▶ (forgotten)
             never updated

REAL learning:  new data ─▶ [ offline training ] ─▶ new frozen weights ─▶ redeploy
```

**Animate:** Top row plays: request → answer → **"forgotten"** fades out. Then the bottom row builds separately as a distinct offline path.

**Speaker notes:**
"Here's a question everyone asks: does the model learn from me while it's running? No. Once training ends, the weights — including the embeddings — are frozen, and inference is strictly read-only. Every call is stateless: the model runs forward, produces an answer, and forgets you completely. But doesn't it remember things within a conversation? It seems to — but that's not the model learning. The client is re-sending the whole history back into the prompt each turn, so the knowledge lives in the context window, not the weights. That's called in-context learning, and it vanishes the moment the context clears. For the model to *actually* learn something new, you run a separate training or fine-tuning pass offline, which produces a new set of frozen weights that you then redeploy. Even 'memory' features in products like ChatGPT don't change the model — they just save facts in an external database and paste them back into the prompt next time. The weights never move during serving. And that's deliberate: it keeps the model stable, reproducible, and safe from being poisoned by whatever users type."

---

## 🔄 Slide 8 — Training vs Inference — Side by Side

**Title:** Two Phases, Opposite Priorities

**On-slide:** (table)

| Aspect | 📚 **Training** | 💬 **Inference** |
|---|---|---|
| Goal | **Learn** | **Apply** |
| Weights | **Updated** | **Frozen** |
| When | Rare | Every request |
| Cost | Big one-time | Forever |
| Who | Researchers | App engineers |

**Visual:** Two columns with the table; 📚 vs 💬 icons on top.

**Animate:** Reveal the table **one row at a time**; color the **Training** column blue and **Inference** column orange so the contrast pops.

**Speaker notes:**
"Let's put them side by side, because almost every property is opposite. Training's goal is to learn; inference's is to apply. Training updates the weights; inference freezes them. You train rarely but infer on every single request. Training chews through huge labeled datasets; inference handles one new input at a time. Training is a massive one-time compute bill; inference is an ongoing, per-request cost that never stops. And they even attract different people — researchers and trainers build the model, while application and platform engineers serve it."

---

## 🏗️ Slide 9 — The Full Lifecycle

**Title:** From Raw Data to Live Answers

**On-slide:**
- Data → Train → Eval
- Deploy → **Serve**
- Monitor → **loop** 🔁

**Visual:**
```
Data ▶ Train ▶ Evaluate ▶ Deploy ▶ Serve(Inference) ▶ Monitor
  ▲                                                     │
  └─────────────── retrain when needed ◀──────────────┘
```

**Animate:** A dot travels around the loop hitting each stage; at **Monitor** the "retrain" arrow glows and sends it back to the start.

**Speaker notes:**
"Zooming out, here's the whole lifecycle. You collect and clean data, train the model to learn its weights, then evaluate whether it's good enough. If it is, you deploy those weights onto servers and start serving inference to real users. You monitor quality in production, and when the data shifts or you want improvements, you loop back and retrain. Training and inference aren't rivals — they're two stages of one continuous cycle."

---

## 💸 Slide 10 — Cost & Hardware

**Title:** Where the Money Goes

**On-slide:**
- Train = **big upfront** 💥
- Infer = **forever** ∞
- Infer often costs **more**

**Visual:** Two bars — Training: one tall spike; Inference: many small bars that keep stacking.

**Animate:** Training spike shoots up fast; then inference's small bars stack one by one until their total **overtakes** the spike.

**Speaker notes:**
"Follow the money. Training is a huge upfront expense — think clusters of GPUs running for days or weeks, sometimes costing millions for frontier models. Inference is cheap per call, but you pay it for every request, forever — so at scale, inference often becomes the *larger* lifetime cost. The engineering focus differs too: training is tuned for throughput, crunching as much data as possible, while inference is tuned for latency, getting one answer back to one user as fast as possible."

---

## 🛠️ Slide 11 — Where AI Engineers Fit

**Title:** Most AI Engineering = Inference-Side

**On-slide:**
- Few build base models
- **Most serve them**
- Fine-tune = a little training
- **← you are here**

**Visual:** A pyramid — narrow top "train base models (few labs)", wide base "serve / build on models (most engineers)".

**Animate:** Narrow top appears first, then the wide base grows underneath and highlights **"you are here"** with a marker.

**Speaker notes:**
"So where will *you* spend your time? Training a large base model from scratch is rare — only a handful of well-funded labs do it. The vast majority of AI engineering happens on the inference side: deploying models, scaling them, designing prompts, building retrieval and agent systems, and optimizing cost and latency. Fine-tuning sits in between — it's a small, targeted round of extra training on top of an existing base model to specialize it. If you're becoming an AI engineer, most of your craft is making trained models useful, fast, and affordable in production."


