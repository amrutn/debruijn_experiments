# ExploreToM state-tracking experiment: methods and reproducibility

Everything needed to reproduce the experiment in `exploretom_tuning/` and to
describe it in a paper. Code paths are given so each statement can be checked.

## 1. Research question

Does fine-tuning a small model to keep a story's state in its local context,
by restating each step and writing the exact world-and-belief state the
question depends on after it (*state-tracking*), improve its accuracy on
theory-of-mind questions more than fine-tuning on a question-specific chain
of the decisive steps (*chain*), or on the answers alone (*direct*)? The
state-tracking trace is local and De Bruijn-structured: each state line is a
function of the previous state line and the steps since it, so answering
never needs to look further back than the last state. To separate that
structure from the two things it is confounded with -- the length of the
trace and the act of restating the whole story -- a *narration* condition is
a length-matched control that is deliberately **not** De Bruijn: it restates
each step and adds a local note of what just happened and who was present, but
carries no running world or belief state forward, so its final line is not a
sufficient statistic and the question can be answered only by integrating the
notes across the whole story. All four exact targets are computed exactly from
the generator's state, so the conditions differ in the format of the reasoning
and in nothing else. An optional fifth condition distils a teacher model's
free-form reasoning.

(Earlier versions also carried a *full ledger* condition, which restated the
whole world state each step and became infeasible at ~13k tokens on the
larger generated stories, and a *segment* condition that decoded the
state-tracking trace one interval at a time; both were removed. Narration was
added after the runs in §13, which therefore do not yet report it. The run
records predate the removals and still name the removed conditions, with
`focused` the former name of `state-tracking`.)

## 2. Benchmark

**ExploreToM** (Sclar, Yu, Fazel-Zarandi, Tsvetkov, Bisk, Choi, Celikyilmaz.
*Explore Theory of Mind: program-guided adversarial data generation for
theory of mind reasoning*, ICLR 2025). Stories are programs over a small
action language: people enter and leave rooms, move objects into containers
or other rooms, tell each other things privately or out loud, and may
witness an action in secret or miss it while distracted. The authors'
`belief_tracker` holds the true world state and every character's first-
and second-order beliefs after each action; questions and answers are read
off that tracker.

**Sources, pinned.**

| Artefact | Location | Version | Digest |
|---|---|---|---|
| Released sample | Hugging Face `facebook/ExploreToM`, `ExploreToM-data-sample.csv` (CC BY-NC 4.0) | revision `a4ac6f257e0034945f829716047ae6306dc625a0` | sha256 `f8462b1a…c448b` |
| Belief tracker | GitHub `facebookresearch/ExploreToM`, `belief_tracker.py` (CC BY-NC 4.0) | commit `6a9372870ddc1c7b9b9ce1f3ee1641dbbdb56760` | sha256 `0a00f850…3b90` |

`exploretom_data.fetch_csv` / `tracker` download both once into `cache/raw`
and verify the digests; the tracker is imported from there, unmodified.

The sample holds 13,309 questions over 619 distinct story structures (the
templated rendering, one sentence per action; an LLM-written *infilled*
rendering of each story is also released but is not used here), generated
adversarially against Llama-3.1-70B-Instruct with 2 to 4 people, 2 to 4
moves, 1 or 2 rooms and at most 15 sentences.

**Generated stories (`DataConfig(dataset='generated')`, `generate_stories`).**
Because the sample saturates (§13), larger stories can be generated locally
with no LLM: `generate_stories.sample_story` drives the belief tracker's DSL
directly, applying a random sequence of valid actions (enter/leave a room,
move an object to a container or a room, tell privately or out loud about a
topic or a location, with optional peeking/distraction) of the requested
size, so the `story_script` is a templated story with an exact state behind
it. The tracker's `QuestionGenerator` gives the first- and second-order
questions with ground-truth answers, emitted in the released CSV schema and
run through the very same `build_problems` verification (parse, replay,
reproduce every answer), so a generation bug drops a story rather than
mislabels it. Sizes are set by the `gen_*` fields (`--dataset generated
--gen-people N --gen-moves N --gen-rooms N --gen-stories N`,
`--gen-interesting-only` to keep only questions whose answer depends on who
is asked); `gen_seed` and the sizes fix the set and are part of its cache
key. Object-state updates and the factual (`memory`/`ground_truth`) question
types are not generated. A 30-story generation at 6 people / 12 moves builds
in seconds with zero parse/replay/verify failures, ~21 steps per story
against the sample's ~7.

## 3. Recovering the states (`exploretom_data`)

The sample does not include the per-step states, so they are recovered:

1. **Parse.** Each templated sentence is matched against the tracker's
   sentence templates (`parse_story`: enter/leave a room, move an object to
   a container or a room, tell privately or out loud about a topic or a
   location, a location declaration). A "While this action was happening,
   … witnessed this action in secret / … got distracted …" sentence
   modifies the action before it; the pair is one *step*.
2. **Replay.** The steps are applied to a fresh tracker (`replay`). The
   story is kept only if the tracker accepts every step and regenerates the
   original text verbatim.
3. **Verify.** The tracker's `QuestionGenerator` regenerates the story's
   questions; the story is kept only if every released question of it comes
   back with the released answer (`regenerate_questions`).
4. **Filter questions.** `memory_before_event` questions ("where was X
   before …") are dropped, their answer not being a function of the final
   state; a question is also dropped if the structured reader of §5 cannot
   answer it from the final ledger state.

| Stage | Count |
|---|---|
| Story structures in the sample | 619 |
| Not parseable (free-text object-state sentences, e.g. "Eric switched the walkie-talkie to a high power mode.") | 117 |
| Refused by the tracker / script mismatch | 0 / 0 |
| Not all released answers reproduced | 4 |
| **Stories kept** | **498** |
| Questions in kept stories | 11,123 |
| Dropped: `memory_before_event` / not readable from the final state | 254 / 8 |
| **Questions kept** | **10,796** |

## 4. Splits

`DataConfig(n_train=1500, n_test=200, q_per_story=4, seed=0)`, `_split`:
the kept stories are put in a seeded order; from each story at most 4
questions are chosen (seeded). The first stories supply the **test set**
until 200 questions are taken; every later story supplies the **training
pool** in order, and the training rows are the pool's first 1,500. Test and
training stories are disjoint.

| Split | Questions | Stories | Question types (reasoning order suffix; `*` = false belief) |
|---|---|---|---|
| test | 200 | 50 | knowledge-2 87 (21\*), knowledge-1 44, room-2 27 (2\*), container-1 15 (1\*), container-2 13 (1\*), memory 8, ground_truth 6 |
| train | 1,500 (pool 1,792) | 375 | knowledge-2 530 (140\*), room-2 314 (7\*), knowledge-1 298, container-2 124 (26\*), container-1 103 (16\*), ground_truth 69, memory 62 |

Test rows by people: 2 → 72, 3 → 56, 4 → 72; by rooms: 1 → 144, 2 → 56.
Answers: yes/no, "knows about it" / "does not know about it", or a container,
room or object name. Stories average 7.3 steps and 78 words (max 178).

## 5. Prompt, targets and grading

**Prompt** (every condition, base model included): the Qwen chat template
with system prompt `exploretom_data.SYSTEM_PROMPT`

> You read a short story and answer a question about it. You may reason
> first. Whatever else you write, the last line of your reply must be the
> answer field: "Answer: <answer>". Answer a yes/no question with yes or no,
> a question that offers two phrases in brackets with one of those phrases,
> and any other question with the name of the container, room or object.

and user turn `Story: <story structure>\n\nQuestion: <question>`.

**Targets** (`exploretom_data.TARGETS`), all exact:

| Condition | Assistant turn |
|---|---|
| direct | `Answer: <answer>` |
| chain | `Tracking the answer to the question through the story.` then, for each step at which the answer to *this* question changes, the step verbatim followed by `-> answer now: <answer after it>`; then the answer field. The running answer is obtained by regenerating the tracker's questions on every prefix (`_running_answers`); factual questions are read from the world state. Mean 1.2 quoted steps. |
| state_tracking (shown "state-tracking") | every step verbatim, each followed by the *question's slice* of the state (`FOCUS_STRIDE = 1`, `--focus-stride`); then the answer field. The slice (`focus_of`) keeps the people the question asks about and the object, container or topic it is about; a line with nothing relevant reads `State: nothing relevant yet.` Its beliefs are written *explicitly* (`FOCUS_BELIEFS = 'explicit'`, `--focus-beliefs`): on every line, each asked person's belief about the asked object ("Samantha believes the bookmark is in the paper bag" / "Samantha does not know where the bookmark is") or topic ("Nicholas has not heard about …"), and for a second-order question the asked person's view of the other ("Nicholas thinks Avery has not heard about …"), whether or not it departs from the truth. This is the local, De Bruijn-structured trace. |
| narration | every step verbatim, each followed by a local `Note:` line describing just that step (`narration_note`): the event -- who entered or left a room, who moved which object where, who told whom what, who announced what -- and, for a witnessable action, who was present in the room, plus any secret watcher or distracted person; then the answer field. The note is **non-cumulative**: it names no running world or belief state, so unlike a state-tracking line the final note is not a sufficient statistic and answering requires integrating the notes across the story. This is the length-matched, non-De-Bruijn control (§1); it isolates the De Bruijn structure from the trace length and the restating. |

With a stride of *k* a state line follows every block of *k* steps, blocks
as even as possible with the earlier ones taking the extra step (`blocks`: 7
steps at stride 2 → states after steps 2, 4, 6, 7; the last state is always
written), so a state line is a function of the previous state line and the
steps since it. The stride and the belief mode are part of the adapter
cache keys.

**State line, departures form** (`state_facts`, `render_state`): `State:
<world>. Beliefs: <departures>.` The world part lists every person's room ("is in the R" /
"has left the R" / "has not entered any room"), every object's container
and room ("the O is in the C in the R"; plus "(it started in the C)" when
the initial location differs from the current one, which is what the
`memory` questions ask), and every container's room. The beliefs part lists
only departures: per person, the objects whose location they believe
wrongly ("X thinks the O is in the C") or do not know ("X does not know
where the O is"), and the topics they have heard about; per ordered pair,
what X thinks Y believes when that differs from X's own belief ("X thinks Y
does not know where the O is", "X thinks Y believes the O is in the C", "X
thinks Y has not heard about T"). Beliefs about people's rooms and about
containers' rooms are not listed (containers never move). When nothing
departs from the truth the line ends `Beliefs: as stated.`

**Sufficiency.** `answer_from_state` answers a question from the structured
content of the *final* state line alone (a container is where the world says;
for beliefs, the explicit form states them outright). Every kept question is
answered correctly by this reader from the state-tracking trace, so the
trace is sufficient for the test set by construction; a row it cannot answer
is dropped at build time.

**Token counts** (Qwen tokenizer, training rows, stride 1, released sample):
prompt mean 206 (max 337); direct target mean 5; chain mean 46 (max 154);
state-tracking (explicit) mean 351 (max 1,454; prompt + target max 1,781);
narration mean 240 (max 515). Narration restates the whole story with a
per-step line, so it is the same order of magnitude as state-tracking (~0.68×
on the sample) and far longer than chain or direct, while carrying no running
state. On the larger generated stories (6 people / 12 moves) the state-tracking
trace averages ~1.0-1.3k tokens and narration ~0.7-0.9k (~0.79×), the operating
point at which the length match matters and where §13's runs separate the
formats.

**Grading** (`extract_answer`, `is_correct`): the text after the last
`Answer` field (first line), lower-cased, with markdown, brackets, a leading
article or preposition and trailing punctuation removed, must equal the
normalised label. A reply without the field is scored on its last line and
counted as unanswered.

**Example** (training row; question "In which container will Samantha
search for the bookmark?", answer "paper bag"), state-tracking trace:

```
Caleb entered the main bookstore floor.
State: nothing relevant yet.
Caleb moved the bookmark to the paper bag, which is also located in the main bookstore floor.
State: the bookmark is in the paper bag in the main bookstore floor. Beliefs: none yet.
Caleb moved the bookmark to the wooden chest, which is also located in the main bookstore floor.
State: the bookmark is in the wooden chest in the main bookstore floor (it started in the paper bag). Beliefs: none yet.
Samantha entered the bookstore's back room.
State: Samantha is in the bookstore's back room; the bookmark is in the wooden chest in the main bookstore floor (it started in the paper bag). Beliefs: Samantha does not know where the bookmark is.
Caleb moved the bookmark to the paper bag, which is also located in the main bookstore floor.
State: Samantha is in the bookstore's back room; the bookmark is in the paper bag in the main bookstore floor. Beliefs: Samantha does not know where the bookmark is.
Caleb told privately to James that the bookmark is in the paper bag. While this action was happening, Samantha witnessed this action in secret (and only this action).
State: Samantha is in the bookstore's back room; the bookmark is in the paper bag in the main bookstore floor. Beliefs: Samantha believes the bookmark is in the paper bag.
Answer: paper bag
```

For the same row, the chain target is the single "told privately" step with
`-> answer now: paper bag`, and the direct target is the answer line alone.
The **narration** target restates the same steps but replaces each state line
with a local, non-cumulative note:

```
Caleb entered the main bookstore floor.
Note: Caleb is now in the main bookstore floor; present there: Caleb.
Caleb moved the bookmark to the paper bag, which is also located in the main bookstore floor.
Note: Caleb put the bookmark into the paper bag in the main bookstore floor; present there: Caleb.
Caleb moved the bookmark to the wooden chest, which is also located in the main bookstore floor.
Note: Caleb put the bookmark into the wooden chest in the main bookstore floor; present there: Caleb.
Samantha entered the bookstore's back room.
Note: Samantha is now in the bookstore's back room; present there: Samantha.
Caleb moved the bookmark to the paper bag, which is also located in the main bookstore floor.
Note: Caleb put the bookmark into the paper bag in the main bookstore floor; present there: Caleb.
Caleb told privately to James that the bookmark is in the paper bag. While this action was happening, Samantha witnessed this action in secret (and only this action).
Note: Caleb spoke privately to James that the bookmark is in the paper bag; secretly watched by Samantha.
Answer: paper bag
```

Every note describes only its own step; none carries the running state
forward. The final note records that Samantha secretly saw the "told
privately" step but says nothing about where the bookmark then is or what
Samantha believes, so the answer follows only from integrating the notes --
in contrast to the state-tracking trace, whose final line states
"Samantha believes the bookmark is in the paper bag" outright.

**Distill (optional).** `traces.py`: the teacher (default Qwen3-32B,
revision `9216db57…`, run locally in bf16 with thinking off, greedy then one
sampled retry; or an Anthropic model through the API) answers the training
question under the student's prompt plus a request for at most 150 words of
plain prose; the trace is kept if its normalised answer equals the label,
else the row is recorded as FAILED and replaced by the next pool row, so the
condition also trains on 1,500 rows (not exactly the rows of the other
conditions). `run_experiments.py --conditions base,direct,chain,state_tracking,narration,distill`
builds the file itself when it is short.

## 6. Model and fine-tuning

| Setting | Value | Where |
|---|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa798…` (`--base-model Qwen/Qwen2.5-1.5B` @ `8faed761…` for the pretrained model) | `train_eval.BASE_MODELS` |
| Precision | bfloat16 weights, activations and LoRA | `TrainConfig.dtype` |
| Attention | torch SDPA; training restricted to the flash/efficient/math kernels (cuDNN excluded, §9) | `ModelConfig.attn_implementation`, `TrainConfig.sdpa_backends` |
| Adapter | LoRA r = 32, α = 64, dropout 0.05, on q/k/v/o/gate/up/down projections of every layer (~36.9M trainable, 2.4%) | `ModelConfig` |
| Loss | next-token cross-entropy on the assistant tokens only (prompt masked with −100); the `<\|im_end\|>` closing the turn is a target | `train_eval.encode_example` |
| Optimiser | AdamW, lr 1e-4, weight decay 0, β defaults, grad-norm clip 1.0 | `TrainConfig` |
| Schedule | cosine to 0 over the run, 3% linear warm-up | `TrainConfig` |
| Batch | 8 sequences × 4 accumulation = 32 per optimizer step | `TrainConfig` |
| Budget | 3 passes over the 1,500 rows = 4,500 samples, 141 optimizer steps; reshuffled each pass | `run_experiments.PASSES`, `train_cfg` |
| Max sequence | 4,096 tokens (no training example exceeds it) | `TrainConfig.max_seq_len` |
| Seed | 0 (LoRA init and sample order); one seed per condition | `run_experiments.SEEDS` |
| Gradient checkpointing | on | `train_adapter` |

## 7. Evaluation

Greedy decoding (`EvalConfig.do_sample=False`), up to 2,048 new tokens,
stopping at `<|im_end|>` / EOS, batched left-padded with prompts sorted
longest first (batch 64; the batch size is not part of any cache key, and
greedy decoding is batch-invariant). A reply that hits the cap is scored
wrong and counted in `capped`. Reported per condition: accuracy over the 200
rows (binomial SEM), *story accuracy* (test stories with all their questions
right), `answered`, `capped`, mean generated tokens, accuracy per question
type and for false-belief versus true-belief/factual questions.

**Learning curve.** While an adapter trains it is evaluated
`TrainConfig.curve_evals = 20` times, evenly spaced over the optimizer steps
with the last at the end of training, on the run's test rows under the same
decoding settings as the final evaluation, so the last curve point equals the
final score. `curve_evals` is part of the adapter cache key (the curve is
produced during training), so changing it retrains the adapters. RNG state is
restored after each evaluation; the curve is stored in the adapter's
`done.json` with the test and decoding specs it was scored under, and in
every checkpoint. The untuned model is evaluated once and is the zero-sample
point of every curve and the dashed horizontal line of both figures.

## 8. Outputs

By default the experiment is run on **both** datasets (the generated
"positive" set and the released-sample "negative" baseline, §11) and a
separate set of figures is written for each; the generated set's file names
carry a `_gen-p<N>m<N>r<N>` suffix so the two never overwrite each other.

`figures/exploretom_accuracy_<model>[_<teacher>][_gen-...].{pdf,png}`: accuracy
per condition (bars, binomial SEM, dashed baseline; a distinct colour per
condition). `figures/exploretom_curve_<model>[_<teacher>][_gen-...].{pdf,png}`:
accuracy against training samples, one line per fine-tuned condition from the
untuned model at zero samples, the untuned model's accuracy dashed. Both use
the compact 3×2.5-inch style of the knockout figures in `../benchmarks`
(14 pt axis labels, 12 pt ticks, 7 pt frameless legend, a faint grid, and no
title). The console prints, for each dataset, the table, the pairwise
differences with their standard errors, the per-type table and the curve
values.

Cache (`cache/`): `raw/` (the CSV and the tracker), `datasets/` (the parsed,
verified problem set with every state line), `adapters/` (LoRA +
`done.json`), `decodes/` (every completion), `evals/` (scores with every
completion and its parsed answer). Keys are SHA-1 of the configs, so a
changed setting recomputes only what depends on it.

## 9. Robustness and numerical safety

Training checkpoints every 20 optimizer steps (adapter, optimizer, schedule,
RNG, counters, curve) and resumes from them; decodes append each finished
batch to a partial file and resume; a unit that fails is reported and the
others continue (exit status 1). A non-finite loss or gradient norm aborts
the unit immediately: on Hopper/Blackwell GPUs cuDNN's fused-attention
backward has produced non-finite gradients in bf16 with padded batches in
several torch releases, which is why training excludes that kernel;
`python check_train.py` reproduces the first optimizer steps under each
kernel on a machine and reports which train cleanly.

## 10. Software

Python ≥ 3.10, `torch`, `transformers` (5.x; tested with 5.16), `peft`
(0.20), `tqdm`, `numpy`, `matplotlib`; `anthropic` (1.x) only for an API
teacher. The tracker needs only the standard library. The default runs both
datasets — about ten units (two × [1 base + 4 conditions]) with twenty curve
evaluations each — which is several hours on one GPU (H100/B200 class);
`--dataset sample` or `--dataset generated` runs a single dataset. A CPU smoke
test (`--n-train 8 --n-test 8 --q-per-story 2` with a small model) exercises
every path.

## 11. Commands

```bash
cd exploretom_tuning
python exploretom_data.py --show 2                 # build + verify the problem set, print examples
python check_train.py --device cuda:0              # optional kernel check
python run_experiments.py --devices cuda:0         # BOTH datasets (generated + sample), default conditions
python run_experiments.py --devices cuda:0 --dataset sample     # only the released-sample baseline
python run_experiments.py --devices cuda:0 --conditions base,direct,chain,state_tracking,narration,distill
python generate_stories.py --people 6 --moves 12 --stories 20 --show 1           # preview generated stories
python run_experiments.py --devices cuda:0 --dataset generated  # only generated (default 6 people / 12 moves / 500 stories)
python run_experiments.py --devices cuda:0 --dataset generated --gen-interesting-only  # false-belief-heavy
python run_experiments.py --plot-only              # figures from the cache
python analyze_ledger.py --stride 1               # where the state-tracking model's state lines go wrong
```

The stride and the belief mode are part of the adapter cache keys, and
figures made with non-default settings carry a `_fs<n>` suffix.

## 13. Run record

**Run 1 (2026-09-06; both ledgers at stride 2, focused beliefs as
departures; everything else as above).** Test accuracy over the 200 rows,
one seed: base 0.485 (only 56% of its replies had an answer field), direct
0.810, chain 0.910, ledger 0.750, focused 0.845; story-level accuracy 0.12,
0.56, 0.70, 0.50, 0.58. On the 25 false-belief rows: base 0.24, direct
0.28, chain 0.76, ledger 0.04, focused 0.44. Chain minus direct was +0.100
(± 0.034), focused minus ledger +0.095 (± 0.040), ledger minus direct
−0.060 (± 0.041).

`analyze_ledger.py` on the cached completions: the ledger model reproduced
the world part of the final state line in 95% of rows but the beliefs part
in only 24.5%, and on the 6 false-belief rows where the beliefs part was
exactly right it still answered with the truth (0 of 6). The focused model
reproduced the beliefs part in 73.5% of rows; on false-belief rows it was
right in 8 of the 10 rows where the beliefs part matched, in 0 of the 6
where it *omitted* the departure clause (writing "Beliefs: as stated"), and
in 3 of the 9 where it wrote a wrong clause. These findings motivated the
explicit belief form of the focused ledger, the return to stride 1 and the
attention-knockout evaluation.

**Run 2 (2026-09-06; both ledgers at stride 1, focused beliefs explicit).**
Test accuracy over the 200 rows, one seed: base 0.485, direct 0.810, chain
0.910, ledger 0.775, focused 0.910; story-level 0.12, 0.56, 0.70, 0.48,
0.74. On the 25 false-belief rows: base 0.24, direct 0.28, chain 0.76,
ledger 0.20, focused 0.76 -- the explicit belief encoding raised the focused
ledger from run 1's 0.44 to 0.76 and its overall accuracy from 0.845 to
0.910, level with chain (focused − chain = 0.000 ± 0.029), while the full
ledger stayed below direct (ledger − direct = −0.035 ± 0.041).

A since-removed attention-knockout evaluation (run under the same adapters)
showed the ledger recovering to its unrestricted accuracy only when the
attention window reached ~512 tokens, roughly its whole reply, and both
ledgers collapsing under narrow windows largely by capping -- repeating
steps once they could no longer see how far they had restated. Because that
confound (an untrained mask) could not be separated from genuine
insufficiency, the knockout was replaced by the trained **segment**
condition (§5, §7), whose result belongs to a later run.

**Run 3 (2026-09-06; generated data, 500 stories x 6 people / 12 moves / 3
rooms, seed 0; 1,500 train and 200 test questions from disjoint stories).**
The harder data separates the formats, the point of generating it. Test
accuracy, one seed: base 0.535, direct 0.610, chain 0.635, focused 0.840;
focused − chain = +0.205 (± 0.043), focused − direct = +0.230 (± 0.043),
chain − direct = +0.025. On the 78 false-belief rows: focused 0.86, chain
0.62, direct 0.54, base 0.55. Two conditions did not yield a usable number
in this run and were fixed afterwards: the full **ledger** crashed because
all its examples exceed `max_seq_len` at this size (see Full ledger size),
and **segment** read 0.000 because the interval decoder only stopped when the
model volunteered an answer, which never happened on ~22-step stories; the
decoder was rewritten to run the story's block count of intervals and then a
readout (`generate_segmented`), and the segment target version was bumped so
its stale decode is refreshed on rerun.

**On difficulty.** Chain and focused tie at 0.910 on the sample and only the 25
false-belief rows discriminate the formats, so this operating point cannot
rank the good formats or, on its own, show whether the segment condition
preserves accuracy. The `dataset='generated'` path (§2, `generate_stories`)
produces larger stories locally for a difficulty sweep -- vary
`--gen-people` / `--gen-moves`, or `--gen-interesting-only`, and read the
formats off `exploretom_accuracy_..._gen-pNmNrN` / the curve as difficulty
rises; the segment/locality comparison is informative where the formats
separate.

## 12. Limitations to state

The stories are the templated rendering, not the infilled prose, so
the result is about state tracking over unambiguous text. The sample was
selected adversarially against Llama-3.1-70B, not against the student, and
after fine-tuning the good formats reach ~0.91, so the operating point is
near-saturated and does not separate them (see the run record). The 117
stories with object-state updates are excluded, so the question mix has no
object-state questions. Evaluation is in-distribution (held-out stories of
the same generator); a transfer test (e.g. ToMi) would need its own data
loader. Answers with more than one valid surface form are graded by exact
normalised match only.
