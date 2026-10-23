# NCM adaptive-depth study — results log

Model: authors' TinyStories checkpoint (generator 88.6M, VQ-VAE 36M), levels
[1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128], 11 generator passes (NFE) per 128-token story.
Sampling: paper defaults (temp 1.0, top_k 0, top_p 1.0, no CFG), level 0 from unigram prior.
Quality metric: gen-PPL (lower = more fluent) under GPT-2 small unless stated; paper uses GPT-2 large.
All runs on laptop CPU, 20–21 Oct 2026.

## 1. Early exit (stop after level k) — N=256, `early_exit.py` + `analyze.py`

| passes k | genPPL | tokens differing from final |
|---|---|---|
| 3 | 234 | 80% |
| 6 | 148 | 44% |
| 7 | 66 | 21% |
| 8 | 54 | 16% |
| 9 | 44 | 12% |
| 10 | 37 | 9% |
| 11 (full) | 20.5 | 0% |

Non-monotone: PPL *rises* from k=0 to k=3 (coarse decodes are garbage, but k=0 garbage is
repetitive, which GPT-2 scores as "predictable"), then falls. The final level alone takes PPL
37 → 20.5. No plateau, so there is nothing to exit early into.

Per-sample stop rules replayed on the logs (decoder confidence ≥ τ, tokens-changed ≤ τ,
generator confidence ≥ τ): best case saves 0.7 passes for +3% PPL; saving 2 passes costs
+80–100% PPL. Adaptive rules do beat uniform depth at matched mean NFE (curve sits below the
fixed-depth curve), but the whole curve is steep. **Early exit is not viable on this model.**

## 2. Level skipping (drop middle levels at inference) — N=256, `skip_levels.py`

| schedule | NFE | genPPL |
|---|---|---|
| full | 11 | 20.5 |
| drop10 / drop5 / drop8 | 10 | 21.5 / 21.7 / 21.8 |
| drop8-10 / drop5-10 | 9 | 23.3 / 23.9 |
| drop5-8-10 | 8 | 26.6 |
| drop11 (last level) | 10 | 38.4 (N=64) |
| drop1-3-5-7-9 | 6 | 143 (N=64) |

One skipped middle level: +5% PPL. Two: +15%. Three: +30%. Five: 7×. Last level untouchable.
Generator degrades gently under a schedule it was never trained on.

### 2b. Level skipping, 5 seeds x 128 samples (released generator, GPT-2 small)

| schedule | NFE | genPPL mean ± sd | vs full |
|---|---|---|---|
| full | 11 | 20.30 ± 0.60 | — |
| drop10 | 10 | 21.27 ± 0.74 | +4.8% |
| drop5 | 10 | 21.83 ± 0.51 | +7.5% |
| drop8 | 10 | 21.88 ± 1.12 | +7.8% |
| drop5-10 | 9 | 23.27 ± 0.90 | +14.6% |
| drop1-3-5-7-9 | 6 | 138.0 ± 6.2 | 6.8× |

Paired by seed (same level-0 draws), drop10 is worse than full on 5/5 seeds. The skip cost is
small but real; it is not noise. Entropy unchanged (3.97–4.00) except the 6-pass schedule (3.89).

## 3. Oracle: where does the information live? — 512 real validation stories, `oracle_levels.py`

Reconstruction accuracy (token exact match) decoding true VQ codes from level subsets:

| subset | #levels | oracle acc |
|---|---|---|
| all 12 | 12 | 99.98% |
| drop any single level (incl. last) | 11 | ≥ 99.0% |
| drop 5,10 | 10 | 99.87% |
| drop 3,5,7 | 9 | 99.17% |
| drop 2,4,6,8,10 | 7 | 98.95% |
| drop 1,3,5,7,9 | 7 | 98.05% |
| levels 7..11 only | 5 | 98.42% |
| keep 0,7,11 | 3 | 94.92% |
| keep 0,4,8,11 | 4 | 91.13% |
| level 11 only | 1 | 79.32% |
| prefix 0..7 | 8 | 94.05% |

The residual stack is massively redundant: every other level can be thrown away and the
tokenizer still reconstructs ~99% of tokens. Yet the *generator* loses 7× PPL on that same
subset (drop1-3-5-7-9). The bottleneck is not information; it is that the generator was trained
on one fixed schedule and cannot consume a sparser one.

Second observation: the generator's last pass matters far more than the tokenizer's last level
(drop11: generator 38 PPL vs oracle 99.0% acc). The fine passes act as *refinement* of coarse
generator errors, not just as information carriers (cf. early-exit: 9% of tokens still wrong
after pass 10).

## Conclusion so far

- Authors' open problem "input-dependent depth" → on TinyStories, inference-only adaptive depth
  does not work; the hierarchy is used as an iterative refiner and all passes are needed.
- The tokenizer is ~2× over-provisioned in levels. A generator trained with **random level
  dropout** (predict level j from any subset of coarser levels) should be able to run 6–7 passes
  at ~full quality, per the oracle numbers. That is the experiment worth a GPU.

## Open / next
- Paper-protocol scoring (GPT-2 large) of table 1: running.
- Repeat skip + oracle on OWT checkpoint (32 levels, 1024 tokens): more room, needs GPU.
- Level-dropout fine-tune of the released generator (training loop not released; write our own
  around `NextConceptTransformer.forward` + `vq.idx_to_ncm_input`).

## 4. Level-dropout fine-tune, round 1 (naive teacher forcing) — vast RTX 4090, 21 Oct

`train_dropout.py`: continue training the released generator for 4000 steps (bs 64, lr 1e-4 cosine,
AdamW 0.9/0.95, wd 0.1, bf16), story-aligned windows from the train stream. Two runs, identical
except level dropout: **drop03** (each input level 1..10 dropped with p=0.3) and **drop00** (p=0, control).
Scored with `skip_levels.py` + GPT-2 **large** (paper protocol), 3 seeds x 256 samples each.

Reproduction check: released/full = **16.15 ± 0.17** gen-PPL, entropy 3.98. Paper Table 9
(block-diagonal, TinyStories): 15.41, entropy 3.98. Pipeline reproduces the paper within 5%.

| schedule | NFE | released | drop00 (control) | drop03 (dropout) |
|---|---|---|---|---|
| full | 11 | 16.15 ± 0.17 | 39.99 ± 1.20 | 36.09 ± 0.91 |
| drop10 | 10 | 17.05 (+6%) | 40.64 (+2%) | 37.47 (+4%) |
| drop5 | 10 | 17.86 (+11%) | 51.56 (+29%) | 36.79 (+2%) |
| drop8-10 | 9 | 18.46 (+14%) | 43.30 (+8%) | 38.53 (+7%) |
| drop5-8-10 | 8 | 21.68 (+34%) | 58.66 (+47%) | 40.40 (+12%) |
| drop2-5-8-10 | 7 | 30.54 (+89%) | 144.7 (3.6x) | 50.70 (+40%) |
| drop2-4-6-8-10 | 6 | 79.72 (4.9x) | 287.0 (7.2x) | 64.05 (+77%) |
| drop1-3-5-7-9 | 6 | 112.7 (7.0x) | 416.9 (10.4x) | 73.00 (+102%) |

(% = cost relative to the same model's own full schedule.)

- **Mechanism works.** Same recipe, only dropout differs: drop03 beats drop00 at every schedule;
  its cost of going from 11 to 8 passes falls from +47% to +12%, and 11 -> 6 passes from 7x to +77%.
- **Recipe is broken.** Both fine-tunes generate 2.2-2.5x worse text than the released model at full
  depth, although their teacher-forced CE is *lower* (clean inputs: released 1.358, drop00 0.927,
  drop03 1.077). Lower CE + worse samples = exposure bias. Samples show doubled words
  ("sliding sliding", "for for") and name drift.
- **Cause (paper §3.4, App. F.3):** the authors train with *conditioning corruption*: a fraction of
  input codes is replaced uniformly from the same level's codebook (rate decreasing linearly over
  levels, mean 0.5), targets kept clean. Our round 1 trained on clean inputs, so the model learned to
  trust inputs it never gets at sampling time. The released recipe: 50k steps, batch 2048, lr 3e-4 ->
  3e-5, 500 warmup (App. C.2, Table 6).

## 5. Round 2: paper recipe (conditioning corruption 0.75 -> 0.25) + level dropout — 21 Oct

3000 steps, same settings as round 1 plus `--corrupt 0.75,0.25`. c00 = corruption only, c03 = + dropout 0.3.
GPT-2 large, 3 seeds x 256.

| schedule | NFE | released | c00 | c03 |
|---|---|---|---|---|
| full | 11 | 16.15 | 16.85 | 16.86 |
| drop8 | 10 | 17.21 | 18.11 | 17.86 |
| drop8-10 | 9 | 18.46 | 19.44 | 18.97 |
| drop5-8-10 | 8 | 21.68 | 23.05 | 22.51 |
| drop2-5-8-10 | 7 | 30.54 | 49.67 | 30.82 |
| drop2-4-6-8-10 | 6 | 79.72 | 310.9 | 54.64 |

- Corruption fixed the round-1 collapse: full-depth quality back to within 4% of released.
- Level dropout does **not** beat the released model at 8-10 passes and only ties at 7. It wins
  only at 6 passes (55 vs 80), where quality is still 3.4x worse than full depth.
- **Verdict: negative.** No schedule gives >=40% fewer passes at equal quality. On TinyStories the
  released generator is already about as skip-tolerant as a dropout-trained one; the extra passes
  are doing real refinement work. Worth a short negative-result note, not a paper.


## 6. Joint tokenizer + generator fine-tune (paper App. I open problem) — 21 Oct

`train_joint.py`: same recipe as c00 (corruption 0.75 -> 0.25, 3000 steps, bs 64, gen lr 1e-4) plus the
tokenizer encoder + decoder trainable (lr 2e-5, reconstruction CE + VQ commitment, codebooks frozen).
j1 adds a coupling loss (weight 1.0): each level's encoder residual is pulled toward the generator's
expected code embedding (generator detached). j0 = joint, no coupling. GPT-2 large, 3 seeds x 256.

| run | trains | coupling | CE @3000 | gen-PPL full (s1/s2/s3) | mean |
|---|---|---|---|---|---|
| released | - | - | 1.358 | - | 16.15 |
| c00 | generator | - | 1.344 | 16.65 / 16.69 / 17.21 | 16.85 |
| j0 | gen + tokenizer | off | 1.350 | 16.71 / 17.01 / 17.13 | 16.95 |
| j1 | gen + tokenizer | on | 1.350 | 16.61 / 16.94 / 16.93 | 16.83 |

- **Tie.** j1 - c00 = -0.02 (per seed -0.04 / +0.25 / -0.28). Joint training at this scale neither
  helps nor hurts; tokenizer reconstruction stays 99.98%.
- Coupling beats uncoupled joint on 3/3 seeds, but only by 0.12 PPL (0.7%): too small to claim.
- Coupling loss fell 0.237 -> 0.153 (the encoder does move toward predictable codes) without making
  the generator's job measurably easier.
- Three bugs on the way, each worth noting for anyone repeating this: tokenizer dropout must be off
  while extracting target codes (else CE 2.15); the tokenizer must encode in fp32, bf16 flips
  fine-level codes (CE 1.87); and the conditioning corruption is mandatory (section 4).
- Prior art: LARP (ICLR 2025, video) and REPA-E (ICCV 2025, images) show joint training helps in
  vision when collapse is prevented; no published NCM/text attempt found.


## 7. Own-mistakes corruption (paper App. F.3 future work) — 21 Oct

Paper App. F.3: "Corruption based on the model's actual errors may better match inference conditions;
we leave this to future work." Tested with `train_dropout.py --self_mix f`: same recipe as c00
(uniform corruption 0.75 -> 0.25, 3000 steps), but a fraction f of the corrupted input codes come from
the model's own guesses (one no-grad clean teacher-forced pass, sampled at temperature 1).
Own guesses are wrong ~47% of the time (logged every 100 steps).

| run | own-guess share | CE @3000 | gen-PPL full (s1/s2/s3) | mean | entropy |
|---|---|---|---|---|---|
| released | - | 1.358 | 15.99 / 16.32 / 16.15 | 16.15 | 3.980 |
| c00 | 0 (uniform only) | 1.344 | 16.65 / 16.69 / 17.21 | 16.85 | 3.976 |
| s50 | 0.5 | 1.376 | 21.11 / 21.79 / 22.57 | 21.82 | 4.000 |
| s100 | 1.0 | 1.387 | 38.49 / 40.78 / 41.25 | 40.17 | 4.047 |

- **Clear negative, dose-dependent.** Replacing random corruption with the model's own guesses makes
  samples worse: +30% at half, 2.4x at all (as bad as no corruption at all, drop00 = 40.0).
- Caveat: own guesses are right about half the time, so the effective wrong-input rate drops from
  ~0.5 (c00) to ~0.37 (s50) and ~0.24 (s100). The paper's Fig. 13 shows lower corruption rates hurt,
  so part or all of the damage may be the lower effective rate. A uniform-corruption control at the
  matched effective rate would separate "fewer wrong inputs" from "realistic wrong inputs".
- Either way, the naive version of the authors' suggested future work does not help in fine-tuning.


## 8. Re-encode refinement sampling on the RELEASED model — 21 Oct (first win over the released sampler)

`sample_variants.py`: generate the story normally (11 passes), decode it, re-encode the decoded tokens
with the tokenizer, keep re-encoded levels 0..k-1 and regenerate levels k..11 (11-k+1 extra passes).
No training; Stanford's weights untouched. GPT-2 Large, 3 seeds x 256 (paper protocol, temperature 1.0).
Diversity: dist-n = distinct n-grams across all samples; xsamp4 = share of a sample's 4-grams found in
other samples (lower = more diverse); dup% = adjacent doubled words.

| sampler | NFE | gen-PPL | sd | entropy | dist-3 | xsamp4 | dup% |
|---|---|---|---|---|---|---|---|
| released (full) | 11 | 16.15 | 0.17 | 3.980 | 0.749 | 0.127 | 0.40 |
| r9 | 14 | 15.13 | 0.15 | 3.980 | 0.747 | 0.125 | 0.33 |
| r8 | 15 | 14.76 | 0.31 | 3.980 | 0.745 | 0.128 | 0.28 |
| r7 | 16 | 13.84 | 0.33 | 3.974 | 0.740 | 0.129 | 0.24 |
| **r6** | 17 | **13.36** | 0.12 | 3.974 | 0.740 | 0.134 | **0.24** |
| r5 | 18 | 12.90 | 0.47 | 3.975 | 0.731 | 0.137 | 0.31 |
| r4 | 19 | 12.90 | 0.31 | 3.976 | 0.727 | 0.142 | 0.33 |
| r3 | 20 | 13.24 | 0.15 | 3.969 | 0.720 | 0.145 | 0.39 |
| t0.8@6+r6 | 17 | 12.96 | 0.17 | 3.969 | 0.739 | 0.134 | 0.23 |
| temp 0.95 | 11 | 15.13 | 0.27 | 3.980 | 0.741 | 0.132 | 0.36 |
| temp 0.9 | 11 | 14.64 | 0.15 | 3.979 | 0.735 | 0.139 | 0.37 |
| temp 0.85 | 11 | 13.75 | 0.41 | 3.968 | 0.723 | 0.144 | 0.33 |
| temp 0.8 | 11 | 13.34 | 0.25 | 3.968 | 0.719 | 0.149 | 0.35 |
| temp 0.7 | 11 | 12.59 | 0.18 | 3.962 | 0.699 | 0.170 | 0.37 |

- **r6 beats the released sampler by 17%** (16.15 -> 13.36, every seed) at essentially the same unigram
  entropy (3.974 vs 3.980), the paper's own diversity check.
- **Not just lower temperature.** At matched diversity, refinement sits below the temperature frontier:
  r6 vs temperature at the same dist-3 (~15.0) is ~11% better; r5 ~10%, r7 ~8%, r8 ~5%, r9 ~4.5%; r3 only ~2%.
  Same picture with cross-sample overlap.
- **Mechanism:** doubled words drop 40% (0.40% -> 0.24%) with r6/r7; temperature does not change them.
  Re-encoding projects the generated story back onto codes the tokenizer actually produces for real text,
  so the regenerated fine levels start from a consistent plan.
- **Cost:** 17 passes instead of 11 (+55% passes; regenerated levels 6-11 hold 520 of 632 code positions).
  Still far fewer steps than AR (128) or diffusion LMs.
- Still to check: other datasets (LM1B checkpoints are released), equal-compute baselines, multiple rounds.


## 9. Re-encode refinement generalizes: LM1B small and large — 21 Oct

`refine_eval.py` runs the authors' exact sampler and decoder (net.generate + eval_shard pick_tokens with
the LM1B frequency/adjacency penalties); refinement forces the re-encoded codes for levels < k through
net.generate itself (verified: forcing every level reproduces the draft exactly, 8/8). Scored with
`score_flm.py`, a copy of the authors' FLM protocol from scripts/eval_merge.py.

**LM1B small** (FLM protocol, GPT-2 Large, 3 seeds x 256)

| sampler | NFE | gen-PPL | sd | vs released | entropy |
|---|---|---|---|---|---|
| r7 | 16 | 77.76 | 1.72 | -18.0% | 4.335 |
| t0.8@1 | 11 | 79.50 | 1.32 | -16.1% | 4.263 |
| r5 | 18 | 80.22 | 3.28 | -15.4% | 4.335 |
| r6 | 17 | 80.80 | 0.39 | -14.8% | 4.340 |
| t0.85@1 | 11 | 83.01 | 1.84 | -12.4% | 4.275 |
| r9 | 14 | 84.11 | 2.26 | -11.3% | 4.327 |
| t0.9@1 | 11 | 85.43 | 0.55 | -9.9% | 4.289 |
| full | 11 | 94.79 | 2.56 | +0.0% | 4.313 |

**LM1B large** (FLM protocol, GPT-2 Large, 3 seeds x 256)

| sampler | NFE | gen-PPL | sd | vs released | entropy |
|---|---|---|---|---|---|
| r6 | 17 | 62.46 | 1.63 | -22.0% | 4.338 |
| r5 | 18 | 63.45 | 1.15 | -20.8% | 4.343 |
| r7 | 16 | 67.10 | 1.15 | -16.2% | 4.342 |
| t0.8@1 | 11 | 67.29 | 0.94 | -16.0% | 4.286 |
| t0.85@1 | 11 | 67.91 | 1.76 | -15.2% | 4.298 |
| t0.9@1 | 11 | 71.24 | 0.71 | -11.1% | 4.313 |
| r9 | 14 | 75.16 | 1.68 | -6.2% | 4.346 |
| full | 11 | 80.11 | 0.34 | +0.0% | 4.339 |

- Refinement wins on every seed for both sizes, at EQUAL or HIGHER entropy; temperature only lowers
  gen-PPL by cutting entropy (4.26-4.30). Refinement dominates the temperature frontier.
- The gain grows with model size: best -18% (small, r7) and -22% (large, r6).
- TinyStories real-text reference: 8.51 gen-PPL (released 16.15, r6 13.36).


## 10. Phase 1: more rounds, conditional tasks — 21 Oct

**TinyStories, more rounds** (GPT-2 Large, 3 seeds x 256; temperature frontier from section 8, same seeds)

| sampler | NFE | gen-PPL | vs released | entropy | dist-3 | xsamp4 | dup% | vs temperature at same dist-3 |
|---|---|---|---|---|---|---|---|---|
| released | 11 | 16.15 | - | 3.980 | 0.749 | 0.127 | 0.40 | - |
| r6 | 17 | 13.36 | -17% | 3.974 | 0.740 | 0.134 | 0.24 | ~11% better |
| r5 | 18 | 12.90 | -20% | 3.975 | 0.731 | 0.137 | 0.31 | ~10% better |
| r6x2 | 23 | 12.64 | -22% | 3.973 | 0.730 | 0.143 | 0.19 | ~11% better |
| r4+r8 | 23 | 12.34 | -24% | 3.969 | 0.724 | 0.143 | 0.25 | ~10% better |
| r5x2 | 25 | 11.70 | -28% | 3.970 | 0.718 | 0.146 | 0.25 | ~12% better |
| r3+r6+r9 | 29 | 11.17 | -31% | 3.963 | 0.708 | 0.158 | 0.23 | ~14% better |

More rounds keep lowering gen-PPL (human text: 8.51), but part of the extra gain is a diversity trade
(dist-3 -5.5% at three rounds). At matched diversity the frontier shift is a steady 10-14%.

**LM1B large, more rounds** (FLM protocol, 3 seeds): released 80.11 (H 4.339); r6 62.46 (-22%);
**r6x2 56.63 (-29%, H 4.346, more diverse)**; r5x2 59.00 (-26%); r4+r8 61.51; t0.9@6+r6 58.97 (H 4.325).

**XSum** (1000 test docs, greedy as in the paper config; ROUGE-1/2/L, GPT-2 gen-PPL of summaries):
released 34.63 / 12.84 / 27.77, gen-PPL 82.4 -> r3 34.88 / 13.09 / 28.10, gen-PPL 69.5 (-16%).
Small ROUGE gain (+0.25-0.33), much more fluent summaries. Two rounds (r5x2) hurt (34.57 / 12.69 / 27.59).

**WMT14 de-en** (1000 test sentences, greedy): released BLEU 20.51 -> r4 20.45, r7 19.01, r6 16.38.
**Refinement hurts translation.** Sanity check: re-encoding a translation without changing anything already
alters 17 of 32 outputs (XSum 5 of 32, TinyStories 0 of 8): the WMT14 tokenizer's round trip is lossy,
so re-reading injects errors into a task where exact wording is scored.

Scope of the method: helps open-ended generation (TinyStories, LM1B) and summary fluency; does not help
exact-match conditional tasks with a lossy tokenizer round trip.


## 11. Phase 2: distil refinement into the generator — 22 Oct

60k refined stories (r5x2, seed 7) from the released model. Fine-tune 1500 steps, bs 128, lr 3e-5,
corruption 0.75 -> 0.25: dist = 50% refined + 50% real, ctrl = real only. Soups = 0.5 released + 0.5 fine-tune.
GPT-2 Large, 3 seeds x 256 (same seeds as the released 16.15).

| model | 11 passes (s1/s2/s3) | mean | with r6 (17 passes) |
|---|---|---|---|
| released | 15.99 / 16.32 / 16.15 | 16.15 | 13.36 |
| ctrl | 16.69 / 16.89 / 16.75 | 16.78 | 13.79 |
| ctrl soup | 16.31 / 16.84 / 16.15 | 16.43 | 13.61 |
| dist | 16.07 / 16.43 / 15.95 | 16.15 | 13.67 |
| dist soup | 15.96 / 16.27 / 15.77 | 16.00 | 13.46 |

- Distillation helps relative to its own control on every seed (-3.8%: 16.78 -> 16.15), but the
  fine-tuning drift (+3.9% for ctrl even with the gentler recipe) cancels it: net equal to released.
- Best 11-pass model (dist soup) 16.00 vs released 16.15: -0.9%, inside the seed noise. Not a win.
- Only about a quarter of the refinement gain transferred, and refinement on top no longer stacks.
- The drift is the bottleneck for any fine-tuning on this budget (batch 128 vs the paper's 2048).


## 12. Sampler ablations + hot drafts (released checkpoints, no training) — 22 Oct

`run_gpu13.sh` on one RTX 4090 (15 min, ~$0.43). TinyStories GPT-2 Large and LM1B large FLM protocol, 3 seeds x 256,
same seeds as sections 8-10 (full / r6 / r5x2 reproduce exactly: 16.15 / 13.36 / 11.70). Frontier = temperature curve
tau 0.7..1.2 interpolated at the variant's distinct-3 (all three seeds now included; the old tau <= 1 curve dropped a seed
where r6 was MORE diverse than released, so r6's matched-diversity gain is -12.4%, not -9.9%).

| variant | NFE | TS gen-PPL | vs released | dist-3 | xs-4 | vs frontier | LM1B-L gen-PPL | vs released | entropy |
|---|---|---|---|---|---|---|---|---|---|
| released | 11 | 16.15 | - | 0.749 | 0.162 | - | 80.11 | - | 4.339 |
| n6 (no re-encode) | 17 | 16.43 | +1.7% | 0.750 | 0.164 | +2.1% | 80.60 | +0.6% | 4.339 |
| q6 (latent re-quantize) | 17 | 14.29 | -11.5% | 0.741 | 0.170 | -6.8% | 65.78 | -17.9% | 4.313 (frontier -7.7%) |
| r6 (text round trip) | 17 | 13.36 | -17.3% | 0.740 | 0.170 | -12.4% | 62.46 | -22.0% | 4.338 (frontier -20.9%) |
| n5x2 / q5x2 / r5x2 | 25 | 16.13 / 12.94 / 11.70 | -0.2 / -19.9 / -27.6% | | | +3.7 / -6.7 / -12.9% | | | |
| draft 1.1 + r6 | 17 | 14.15 | -12.4% | 0.753 | 0.156 | -14.7% | | | |
| draft 1.2 + t0.8@6 + r6 | 17 | 14.98 | -7.3% | 0.766 | 0.143 | -22.0% | | | |
| draft 1.2 + t0.8@6 + r5x2 | 25 | 12.45 | -22.9% | 0.740 | 0.167 | -18.1% | | | |
| draft 1.2 + r6x2 (LM1B) | 23 | | | | | | 62.47 | -22.0% | 4.367 (dist-3 0.894 vs 0.895, xs-4 0.040 vs 0.043) |

- **The re-encoding carries the gain**: resampling the fine levels from the draft's own codes does nothing.
- **Text round trip > latent re-quantization (Scale-travel analogue)**: q gets about half the matched-diversity gain
  (TS -6.8 vs -12.4%, LM1B -7.7 vs -20.9%). q keeps level 0 72% of the time; r changes it 96%.
- **Hot draft + refine = better than released on every axis**: draft tau 1.1 + r6: -12.4% gen-PPL, dist-3 0.753 > 0.749,
  xs-4 0.156 < 0.162, entropy 3.981, doubled words 0.26% vs 0.39%. Frontier gains with hot drafts 15-22%.
- Codes changed by the round trip (768 drafts, levels 0..11): 96 84 75 71 61 68 65 36 62 70 72 20 %; latent re-quantization
  28 77 69 61 49 61 60 25 60 65 67 11 %. `roundtrip_check.py`: re-encode + decode reproduces 751/768 drafts exactly
  (97.8%, 99.98% of tokens).
- Diversity check on section 11: the distilled model's 16.15 comes at dist-3 0.719 (= tau 0.8, which scores 13.34), so at
  matched diversity distillation is ~21% WORSE than the temperature frontier. Phase 3 (distillation at batch 2048,
  run_gpu12.sh) not run.


## 13. Evidence + tuning run (run_gpu14.sh, box 57230955) — 22 Oct evening

Box self-stopped after ALL_DONE (guard 1 worked); the laptop watcher died (Windows fork failure on low battery), the GPU
was re-rented while stopped, so results were pulled with `vastai copy` stopped-box -> temp box (needs a container
reboot to see the files) -> scp. Run + retrieval cost ~$1.08 (credit $2.45 -> $1.37).

**Held-out seeds 4-6 (TinyStories, GPT-2 Large; frontier = tau 0.7..1.2 on the same seeds):**
released 16.32 (d3 0.748, xs4 0.165) | RT6 13.23 (-18.9%, frontier -14.5%) | RT5x2 11.94 (-26.8%, -15.3%) |
RT3+6+9 11.50 (-29.5%, -14.3%) | t0.8@6+r6 12.98 (-20.5%, -14.8%) | d1.1+r6 13.96 (-14.5%, -13.5%) |
d1.2+t0.8@6+r6 14.46 (-11.4%, -17.7%, d3 0.755) | **d1.2+t0.8@6+r5x2 12.47 (-23.6%, -21.4%, d3 0.742)** |
n6 16.12 (-1.2%) | q6 14.27 (-12.5%, -8.5%). Everything from seeds 1-3 replicates; matched-diversity gains slightly larger.

**Tuning grid (seeds 1-3):** d1.3+t0.8@6+r5x2 12.94 (-19.9%) at d3 0.749 / xs4 0.157 = SAME diversity as released;
d1.15+t0.85@6+r6x3 12.73 (-21.2%) at d3 0.748; d1.25+t0.8@6+r6x2 13.84 at d3 0.764 (-25.9% vs frontier);
d1.3+t0.8@6+r6 15.80 at d3 0.775 = human diversity (-25.3% vs frontier); r5x3 11.38 (-29.6%).

**LLM judge (Qwen2.5-7B-Instruct, 768 paired stories, both orders, P(prefer second-named)):** full vs r6 0.634
[0.617, 0.651] (both orders: r6 30.6% / full 3.3%); t0.8 vs r6 0.506 (tie at equal gen-PPL); full vs d1.1+r6 0.531;
full vs r3+6+9 0.587; t0.8 vs r5x2 0.529 [0.515, 0.542] (equal diversity); full vs t0.8 0.537; q6 vs r6 0.557.
Judge first-shown bias 0.78-0.91 (order-averaged).

**MAUVE vs 768 human openings (GPT-2 Large features):** released 0.882; t0.9 0.946; t0.8 0.943; r6 0.941; r5x2 0.955;
r3+6+9 0.945; t0.8@6+r6 0.946; d1.1+r6 0.929; d1.2+t0.8@6+r6 0.952; n6 0.913; q6 0.925 (sample-set noise ~0.03).

**LM1B medium (seeds 1-3, FLM):** released 87.47; r6 69.43 (-20.6%); r6x2 62.30 (-28.8%, H 4.342); t0.8 70.22 (H 4.285).
Scale trend RT6: small -14.8%, medium -20.6%, large -22.0%. **LM1B large held-out 4-6:** released 79.87; r6 -20.2%;
r6x2 -27.1%; d1.2+r6x2 -20.7% at H 4.368; q6 -16.6% (H 4.312); n6 +1.4%.

**Sudoku-hard (1000 puzzles, greedy, released 25M model):** released 84.1% solved; r9 84.7; r8 84.9; r6 85.5 (14 newly
solved, 0 broken, McNemar p=1e-4); r6x2 85.2; r3+6+9 85.9 (19 / 1, p<1e-4). Round trip changes only 2-20% of codes.

**OWT:** checkpoint not released (hub folder has only TODO.md).


## 14. Run 15: verifier-guided Sudoku search, round-trip guidance, held-out grid winners — 22 Oct ~19:00

Box 57249444 (4090, ~$0.12 incl. setup), destroyed by the watcher; credit $1.25.

**Sudoku-hard "think until correct"** (`sudoku_search.py`, 1000 puzzles; exact checker = Sudoku rules + given clues,
never the solution; clue parsing verified 1000/1000; verified == solved). Solve rate vs mean generator passes per puzzle
(draft = 12 passes incl. the level-0 query):
- resample @ tau 1.0: 84.1@12 -> 89.9@16.8 -> 91.0@19.2 -> **93.4@23.9** (10 attempts)
- resample @ tau 0.5: 90.0@26.4
- refine6 @ 1.0: 89.5@19.4; refine6 @ 0.5: 88.0@20.0; refine4 @ 0.5: 89.9@21.7; refine3 @ 0.5: 90.7@22.4; refine8 @ 0.5: 86.6@17.7
=> With a perfect verifier, fresh resampling beats round-trip refinement at equal compute (retries stay correlated with
the wrong board). Test-time compute lifts NCM Sudoku-hard 84.1 -> 93.4%, but via resampling, not via our method.

**Round-trip guidance** (`g<w>`: regenerate on [canonical; draft] stacks, sample from (1+w) canon - w draft logits),
TinyStories seeds 1-3: r6 13.36 (frontier -12.4%, H 3.974); r6+g0.5 13.63 (-12.0%, H 3.920); r6+g1.0 14.25 (-9.9%,
H 3.841); r6+g2.0 17.31 (-1.5%, H 3.692); r5x2+g1.0 12.55 (-10.5%); hot-draft variants with g1.0 also no better.
=> NEGATIVE: guidance lowers per-sample unigram entropy (more repetitive stories) and never beats plain RT.

**Held-out seeds 4-6, tuning-grid winners** (frontier = run 14's temperature curve on the same seeds; released 16.32,
d3 0.748, xs4 0.165):
- **d1.3+t0.8@6+r5x2: 12.96 (-20.6%) at d3 0.754 / xs4 0.153 = MORE diverse than released; -25.2% vs frontier**
- d1.2+t0.8@5+r5x2: 12.32 (-24.5%) at d3 0.741 / xs4 0.165; -22.1%
- d1.15+t0.85@6+r6x3: 12.71 (-22.1%) at d3 0.745 / xs4 0.161; -20.9%
- d1.25+t0.8@6+r6x2: 13.89 (-14.9%) at d3 0.762; -26.3%
- d1.3+t0.8@6+r6: 15.88 (-2.7%) at d3 0.774 / xs4 0.136 (human: 0.775 / 0.139); -24.5%
- r5x3: 11.35 (-30.5%) at d3 0.720; -17.1%


## 15. Run 16: fairness controls from the design workflow — 22 Oct ~20:00

Box 57255648 (4090), destroyed by watcher; credit $1.02. Design workflow (27 agents) verdict: inference-only ideas near
ceiling; plan + 12 paper fixes in scratchpad plan.txt.

**11-NFE temperature twins (TinyStories, seeds 4-6, GPT-2 Large; same per-level temperatures, NO round trip):**
d1.3+t0.8@6+r5x2 12.96 vs twin t1.3@1+t1@5+t0.8@6 17.99; d1.2+t0.8@6+r5x2 12.47 vs twin t1.2@1+t1@5+t0.8@6 17.10;
d1.2+t0.8@6+r6 (14.46 in run 14) vs twin t1.2@1+t0.8@6 17.73 (replicate d1.2+t0.8@6+n6 17.51); t0.8@6+r6 12.98 vs t0.8@6
15.46. => The gain is the round trip, not the temperature schedule. (Best 11-NFE schedule seen: t0.6@1 12.10, lower
diversity; diversity analysis still to do.)
**LM1B large twins:** d1.2+r6x2 63.37 (H 4.368) vs twin t1.2@1+t1@6 90.27 (H 4.361); r6x2 58.22 vs full 79.87.
**RT on HUMAN text ('h'):** h 8.51; h+t0.8@6+r6 9.21; h+r6 9.39; h+r5x2 10.35; h+r3+6+9 12.12. RT does NOT sharpen human
text below its own gen-PPL => projection toward the model's data manifold, not a sharpener. (It slightly degrades real
text: realisation error of the generator.)
**Sudoku:** plan<k>@T retries never beat fresh resampling (resample@1.0 92.8-93.4% @ ~23.7 NFE; best plan9@1.0
92.3-93.3% @ ~24) => negative. Greedy RT on easy: 99.6 -> 99.7%; on MEDIUM: 96.7 -> 95.9% (r6), 96.0% (r3+6+9) => RT
HURTS medium. Sudoku claim must shrink to "helps on hard only".


## 16. Run 17: self-correction TRAINING pilot (batch 2048, 2x H100 PCIe) — 22 Oct 21:20-22:10, $6.49

Arms (200 steps, batch 2048, lr 1e-4 warmup 30 cosine->0, AdamW 0.9/0.95, wd 0.1, corruption 0.75->0.25):
ctrl = paper recipe (clean targets); bscu = re-quantized self-correction targets (`--bsc`); bscnn = `--bsc --corrupt_nn 16`.
13.6 s/step on H100 PCIe at micro 32 (~151 seq/s). `--smoke_bsc`: corruption off reproduces vae.encode exactly.
Clean val CE @200: ctrl 1.325, bscu 1.275, bscnn 1.138 (released 1.354).
11-pass, seeds 4-6, GPT-2 Large (same box): released full 16.04 (d3 0.745) | t0.9 14.59 (0.728) | t1.1 18.99 (0.759) | r6 13.06
ctrl: full 16.94 (d3 0.753, xs4 0.159), t0.9 15.19 (0.740), t1.1 19.70 (0.767), r6 13.63  => NO DRIFT (about -3..-4% vs
released global-T curve at matched d3): batch 2048 fixes the drift that killed sections 4-7 and 11.
bscu: full 17.13 (0.754), t0.9 15.53 (0.736), t1.1 18.45 (0.763), r6 14.17 => ~tie with ctrl; RT helps it less.
bscnn: full 23.43 (d3 0.806, H 4.043), t0.9 19.13 (0.781), t1.1 30.83 (0.835), r6 19.49 => much more diverse; looks ~-20%
vs the GLOBAL-T curve at d3 0.78, but the released model's best per-level schedule at that diversity is ~18.1
(t1.3@1+t0.7@6: 18.09 at 0.780) => bscnn is ~6% WORSE against the envelope. No training gain yet.
LESSON: the watcher copied pickles only; trained checkpoints were destroyed with the box. Copy ckpt/ before destroy.
Design workflow (21 agents, plan in scratchpad train_plan.txt): best recipe "RT-AMORT" (train fine levels on the model's
own coarse codes toward round-trip targets, KL anchor to released, warm+cold banks, envelope-vs-envelope yardstick),
p_success ~0.15, ~$14 (cap ~$19), ~6 h on 4x RTX 4090.


## 17. Run 18: round trip on an IMAGE next-scale model (VAR-d16, ImageNet 256) — 22 Oct 23:00-00:15, ~$1.05

`var_rt.py` = VAR.autoregressive_infer_cfg verbatim + forcing hook (cfg 1.5, top-k 900, top-p 0.96; 10 scales
1..16). Sanity: forcing all scales to the draft's own codes reproduces 2000/2000 drafts exactly. Image round trip
(decode -> 8-bit image -> VQVAE re-encode) changes 29/58/74/81/88/89/84/90/92/53% of codes per scale (latent
re-quantization q: 26-92%, last scale 20%) => VAR's sampled stacks are non-canonical too.
FID-10k (pytorch-fid features vs the 10k ADM reference images; 10 per class): base 9.133 | r8 8.922 | r7 9.132 |
r6 9.299 | r5 9.354 | q7 9.260 | n7 9.220 (= base distribution, so noise ~0.1) | r7x2 9.249.
=> NO meaningful gain on images (best -0.21 FID at the finest restart, about 2x the noise; deeper restarts worse).
The phenomenon (non-canonical sampled codes) generalises; the benefit of the round trip does not (pixel round trip is
lossy, and VAR conditions each scale on all previous scales). 50k confirmation skipped (Pierre: write up).
Box: vast ssh proxy refused; used direct ssh (public_ipaddr + 22/tcp HostPort).


## 18. Run 19: referee shield (timing, per-sample canonicality, best-of-N, second scorer) — 23 Oct 15:00-18:30, $0.69

Box 57387338 (RTX 4090, Utah). refine_eval.py now records per variant: wall-clock `seconds` with a CUDA-synchronised
split `time` = {gen, enc, dec}; per-sample `canon` = share of the draft's level 0-5 codes that re-encoding changes
(non-canonicality); `draft_texts`/`draft_raw` (the draft before any round); new variant `b<N>` = keep, per sample,
the most canonical of N drafts. New scripts: score_ppl.py (any HF causal LM, bf16, per-sample PPL, text or FLM
protocol), canon_analyze.py (Spearman + quintiles).

TIMING (256 TinyStories, batch 128 / 128 LM1B-large, batch 32; second run after warm-up):
TS: released 1.4 s (5.6 ms/story) | r6 2.7 s (1.9x for 1.55x passes) | d1.1+r6 2.7 s | r5x2 4.1 s (2.9x) |
r3+r6+r9 5.5 s (3.8x). LM1B-large: released 4.5 s (35 ms/seq) | r6 8.3 s (1.8x) | r6x2 12.5 s (2.8x).
Tokenizer encode+decode+pick = ~10% of the total. Wall-clock tracks FLOPs (fine levels are the long passes), not NFE.

CANONICALITY PREDICTS QUALITY (TinyStories, 3 seeds x 256, GPT-2 Large per story):
released sampler: Spearman(canon, log PPL) = +0.37 +- 0.12 (p < 1e-4 every seed); quintiles by canon: 13.6 / 14.4 /
15.0 / 17.7 / 20.1 (least -> most rewritten; seed-3 numbers 12.9..22.3). r6: improves 92% of stories, rho(canon, gain)
= +0.28 +- 0.004; per quintile draft -> refined: 13.6->11.9, 15.1->12.6, 16.0->13.3, 17.8->14.1, 20.1->15.1 (87-96%
improved). => the more the tokenizer rewrites a draft's coarse codes, the worse the draft and the more RT helps it.

BEST-OF-N BY CANONICALITY (TS seeds 1-3): b4 14.52 (-10.1%, canon 0.687) | b8 14.08 (-12.9%, 0.652) | b4+r6 12.07
(-25.3%) | b8+r6 11.90 (-26.3%) | b8+r5x2 10.98 (-32.0%). BUT diversity falls: dist3 0.749 -> 0.711 (b4) / 0.699
(b8) / 0.681 (b8+r5x2), xs4 0.162 -> 0.203 / 0.216 / 0.230 => vs the temperature frontier at matched dist3: b4 and b8
+11.7% WORSE, b4+r6 only -4.5% (r6 alone -12.4%), b8+* below the curve's diversity range. LM1B-large: b4 79.69 vs
80.11 (nothing), b4+r6 61.96 vs r6 62.46 (nothing). => the selector picks the blandest drafts; negative at matched
diversity. Regenerating the fine levels is what helps; selecting drafts by canonicality is not.

SECOND SCORER Qwen2.5-7B (base, bf16; FLM protocol with the first 128 Qwen tokens for LM1B) on the 24 headline pickles:
every relative gain replicates within a few points. TS s1-3: released 12.28, r6 10.09 (-17.8% vs -17.3% GPT-2L),
r5x2 8.67 (-29.4 vs -27.6), d1.1+r6 10.84 (-11.7 vs -12.4). TS s4-6: t0.8 -18.8, r6 -19.1, r5x2 -28.3, r3+r6+r9
-31.8 (vs -29.5), d1.2+t0.8@6+r5x2 -24.1; twins: d1.3 recipe -21.3 vs its 11-pass twin +12.0 (GPT-2L +10.2); human
4.70, human+r6 5.54. LM1B small: t0.8 -11.3 (GPT-2L -16.1), r6 -18.4 (-14.8) = the one place the scorers rank t0.8 vs
r6 differently. LM1B medium: r6 -22.5, r6x2 -31.9 (-28.8). LM1B large: t0.8 -14.7, r6 -22.9 (-22.0); s4-6: r6x2
-30.4 (-27.1), d1.2+r6x2 -24.3, twin +10.8.

BUG: LM1B canon analysis crashed: LM1B `texts` are '' (dec() cuts at the leading [CLS]) so draft_texts were empty.
Fixed: draft_raw stored, score_ppl scores it under FLM. Rerun = run_gpu20.sh (box 57395717), results below when in.
VAST GOTCHA (23 Oct): instances are now created with intended_status=stopped on every host; launch_box.sh issues one
`vastai start instance` instead of destroying (five boxes were killed as STOPPED-ON-CREATE before this was found).
Paper: sections 'Non-canonicality predicts quality', 'Selecting drafts by canonicality is a diversity trade',
'Wall-clock cost', 'A second scorer'; tables canon/bestof/timing/scorer2; 18 pp.

RUN 20 (box 57395717, $0.41; credit $17.35): LM1B-large canonicality, 3 seeds x 256, GPT-2 Large per sequence (FLM):
released: canon 0.859 (range ~0.65-0.97, i.e. nearly every draft heavily rewritten), rho(canon, log PPL) = +0.10 +- 0.09
(n.s. per seed); quintile draft PPL 76.4 / 78.1 / 79.8 / 81.5 / 84.7 -> weak trend. r6: improves 88% of sequences,
rho(canon, gain) +0.09; quintiles draft->refined 76.4->62.0 ... 84.7->65.3. Qwen: b4 +1.4% (worse), b4+r6 -22.8% = r6
-22.9%. => the per-sample signal is strong where non-canonicality varies across drafts (TinyStories 0.46-0.90), weak
where it is uniformly high (LM1B). Paper text qualified accordingly ("On TinyStories, ...").
