# Explained, from scratch

This walks through the whole study: why the question is worth asking, how the experiment was
built, which traps were found along the way, and what the numbers do and do not say. No prior
knowledge of machine learning is assumed. Terms are explained where they first appear, and the
README has a glossary if you want them in one place.

---

## 1. Why "accuracy" is not the number you want

A language model is a program that continues text. Give it a question and a description of a
database, and it continues with a database query. Run that query, and you get an answer.

Here is the part people find surprising: **ask the same question twice and you can get two
different answers.** The model does not look up a fact, it picks words one at a time according
to probabilities, and unless you force it to always pick the single most likely word, there is
randomness in every choice. That randomness is deliberate. Forcing the most likely word every
time — "temperature zero" — makes models repetitive and, on this kind of task, often worse.

So "the model is 42% accurate" is a summary of one roll of the dice per question. It hides the
question a product owner actually has: *if this thing answers a thousand customer questions
tomorrow, how many will be right?*

Two numbers, computed from the same ten answers per question, separate this:

- **pass@10** — at least one of ten attempts is right. This is the familiar benchmark score.
  It answers *"can the model do this at all?"*
- **pass^10** — all ten attempts are right. It answers *"can I depend on it?"*

Nobody sane runs a product on "at least one of ten". But almost every published score is that
shape, because benchmarks grew up around research, where a human reads the output.

The distance between the two is the **reliability gap**. For the untrained model measured here
it is over 20 points: a benchmark score of 41.7% comes with only 19.0% of questions answered
correctly every single time.

The estimators themselves are not new. pass@k is standard in code generation, and pass^k comes
from agent-benchmark work. What is missing from the literature is what happens to *the gap*
when you fine-tune. That is the hole this fills.

---

## 2. The question

**Fine-tuning** means taking a model someone else trained and training it further on your own
examples, so it gets better at one specific job. It is the standard move when a general model
is not good enough at your task, and it reliably raises benchmark scores.

Does it also make the model **dependable**? Two stories, both plausible:

- **It should get worse.** Training on examples is known to reduce variety in a model's output.
  A model that has learned one house style will produce nearly the same answer every time. If
  those answers are right, reliability jumps. If the model's favourite answer is wrong, it will
  be wrong ten times out of ten — and the questions it used to get right occasionally, it now
  never gets right. Capability could rise while the gap widens.
- **It should get better.** Fine-tuning teaches the shape of a correct answer. Much of the
  flakiness in an untrained model is formatting drift and hesitation, not genuine confusion.
  Removing that could close the gap.

The prediction was registered in advance, with the deciding rule written down before the
numbers existed: the headline is the **change in the gap**, with a confidence interval, and the
interval has to exclude zero before the result is called a finding at all.

---

## 3. The experimental design, and the trap in it

The comparison is between two models:

- **F0** — the base model, untouched.
- **F1** — the same model, fine-tuned on 5,851 examples of question → correct query.

Both are then packaged identically and asked the same 496 questions ten times each.

### Why not just compare against the published baseline?

An earlier project already measured this base model on these questions and published the
result. It is tempting to fine-tune, measure the fine-tune, and compare against those published
numbers. That would be a trap, and it is the single most important design decision here.

Between "the published number" and "my number" sits a whole pipeline: the file format
conversion, the compression settings, the serving software and its version, the chat template,
invisible characters in a configuration file. Any of them can move a score by a point or two.
If any of them changed, you would credit the fine-tuning for it.

So the untrained model is pushed through **this project's own pipeline** and measured again.
That is F0, the control. If F0 does not reproduce the published number, the pipeline moved,
not the model.

This turned out to matter more than expected: Ollama, the serving program, **updated itself**
from 0.34.1 to 0.34.2 partway through the study. Without F0, that alone would have been
indistinguishable from an effect of fine-tuning.

### Four ways the pipeline nearly lied

Getting F0 to be the same model as the published one took four fixes. Each was caught by an
automated check rather than by reading code, which is the point: nobody would have spotted any
of these by looking.

1. **The compression recipe is not one recipe.** The format used, `Q4_K_M`, shrinks most of the
   model's numbers to 4 bits but keeps some at 6 bits for quality. *Which* parts get the extra
   bits is decided by a rule inside the conversion tool — and that rule changed between
   versions. The freshly converted model matched the published one on 404 of its 434 internal
   blocks, and the 30 that differed were not wrong, just stored at different precision. The fix
   was to read the published file's layout and force the same choice.

2. **Hidden sampling defaults.** The conversion tool copies a handful of settings out of the
   model's configuration — temperature 0.7, top_p 0.8 and so on — into the model file. The
   published file has none of them. A serving program that reads those settings would have run
   the new models with different randomness from the baseline, producing a difference that
   looked like a result. They are now stripped, and the file comparison checks *every* setting
   rather than a hand-picked list, because a hand-picked list is exactly how this one slipped
   through the first time.

3. **Invisible characters.** Windows silently converts line endings when writing text files.
   Those converted line endings landed inside the **chat template** — the wrapper text that
   surrounds every question. Every single prompt would have differed from the baseline's by
   invisible characters. The build script compares the resulting template with the reference
   and refused the model until it was written correctly.

4. **A dropped flag after merging.** After training, the small adapter file is folded back into
   the model, and the tokenizer files are saved alongside. The library used for saving quietly
   omitted one setting — whether to prepend a special "beginning of text" marker. With the
   setting missing, the serving program's own default decides, and the fine-tuned model could
   have been fed a token the baseline never saw. Tokenizer files are now copied byte-for-byte
   instead of being re-saved.

After those four fixes, F0's file is **byte-for-byte identical** to the published model: all
434 blocks and every setting. And when measured, it reproduced the published scores within
noise. Only then was F1 worth comparing to anything.

---

## 4. Building the fine-tune

### The data, and the thing that ruins studies

Training used 5,851 question-and-query pairs from the BIRD benchmark's training split, covering
66 databases.

The failure that quietly destroys studies like this is **contamination**: a training example
that is also in the test set. The model then looks brilliant because it is being graded on its
homework. The check here is deliberately blunt: no database may appear in both sets, and no
question may appear in both after normalising the text. The build refuses to write any files if
either check fires. On the real data: zero shared databases, zero shared questions.

A second, subtler rule: the validation set — the questions used to decide when to stop training
— is made of **three whole databases** that never appear in training. Holding out individual
questions would not be enough, because a model that has seen a database's structure has an
advantage on every question about it.

Examples longer than 3,072 tokens were dropped, not shortened. Shortening an example teaches the
model to answer a question it was never fully shown. All 383 dropped examples came from a single
database whose structure alone fills about 5,000 tokens.

### The training itself

Training all 3 billion numbers in the model needs far more memory than a home graphics card
has. **QLoRA** solves this in two moves: compress the original model to 4 bits and freeze it,
then train a small set of new numbers bolted alongside — an **adapter** of about 60 million
numbers. Peak memory: 4.1 GB, on an 8 GB card, in 3.5 hours.

One check runs before training starts. The model sees the database structure, the question, and
then writes the answer; only the answer should be trained on. Getting this wrong — training on
the whole text — produces a model that looks trained and is quietly worse. So the first batch is
decoded and the trained-on portion must be exactly the answer and its end marker, or the script
refuses to run.

### Knowing when to stop

Training longer is not better. The model was checked against the held-out databases every 50
steps:

![Training and validation loss](images/training-curve.svg)

The score on questions it studies keeps improving. The score on held-out databases stops
improving at step 200 and then drifts the wrong way. That divergence is **overfitting** —
memorising instead of learning — and it decides which version to keep: the step-200 one, not
the final one. The rest of the run is published rather than deleted, because knowing where a
recipe starts to hurt is part of the result.

One change is visible without any statistics. Asked a question, the untrained model writes
free-form SQL, while the fine-tuned model writes in the dataset's house style: short table
aliases, one line, no trailing semicolon. It learned the *form* of the answers.

---

## 5. How the answers are scored

Every answer is scored by **running it**. The model's query and the official correct query are
both executed against the real database, and the result tables are compared. A query that is
worded differently but returns the right rows counts as right; a query that looks plausible but
returns the wrong rows counts as wrong.

Two questions are excluded because their official query takes longer than 30 seconds to run
against the database — the same two the published baseline excluded. That leaves 496 scored
questions, each answered ten times by each model: 9,920 scored answers.

The measurement code is the earlier project's, used **unchanged and pinned to one commit**. The
wrapper script here refuses to run if that code has been modified, and records the serving
program's version next to the results.

---

## 6. Reading the numbers honestly

Both scores are computed with the standard unbiased estimators, from the same ten answers per
question.

The comparison needs three safeguards:

- **Same questions.** Both arms are scored on the questions both answered, not on each arm's own
  best set.
- **Uncertainty, stated.** Every difference comes with a 95% confidence interval from a
  **bootstrap**: re-draw the 496 questions at random, with replacement, ten thousand times, and
  watch how much the answer moves. If the resulting range includes zero, the honest reading is
  "no detectable change", however suggestive the headline looks.
- **Questions, not attempts.** The bootstrap re-draws questions, because questions are the
  independent unit. Re-drawing individual attempts would produce a far narrower interval and a
  false sense of precision.

The analysis was also written twice — once here, once in the earlier project — and the two
implementations agree to the decimal on the control comparison.

---

## 7. The result

> **Pending.** F1's evaluation is still running. This section will state the change in
> capability, the change in reliability, and the change in the gap, each with its interval,
> followed by what it means and what it does not.

---

## 8. What this does not show

- One model, one size (3 billion numbers), one task (database questions), one training recipe,
  one run. A larger model, a different task, or different training settings may behave
  differently, and this study cannot say.
- The scorer is strict but not omniscient: a query can be right in a way the official answer did
  not anticipate.
- Every answer came from one machine and one serving version, both recorded.

What it does show, whichever way the number falls, is that the two questions — *can it?* and
*can I depend on it?* — have to be measured separately, because a single benchmark score cannot
tell you which one it answered.
