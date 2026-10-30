# Round-Trip Refinement for Next-Concept Models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23737646.svg)](https://doi.org/10.5281/zenodo.23737646)

Preprint and code archive: https://doi.org/10.5281/zenodo.23737646 (Pierre Khoury, King's College London, 28 October 2026).

A training-free sampler for [Next-Concept Modeling](https://arxiv.org/abs/2610.14162) (NCM). Decode the draft, re-encode
the text with NCM's own tokenizer, keep the re-encoded coarse levels and let the unchanged generator redo the fine levels.

```
draft = NCM sample (11 passes) -> decode -> re-encode -> keep levels 0..k-1 -> regenerate levels k..11 (12-k passes) -> decode
```

On the released checkpoints, with the authors' own sampler, token picker and evaluation protocol:

| task | metric | released sampler | round-trip refinement |
|---|---|---|---|
| TinyStories | Gen. PPL (GPT-2 Large) | 16.15 | 13.36 with RT6 (-17%), 11.17 with three rounds (-31%) |
| LM1B large | Gen. PPL (FLM protocol) | 80.11 | 62.46 with RT6 (-22%), 56.63 with two rounds (-29%) |
| XSum | ROUGE-L / summary Gen. PPL | 27.77 / 82.4 | 28.10 / 69.5 (RT3) |
| WMT14 de-en | BLEU | 20.51 | no gain (lossy tokenizer round trip) |
| Sudoku-hard | solved | 84.1% | 85.9% (three rounds; no gain on easier puzzles) |
| VAR-d16 images | FID-10k | 9.13 | no meaningful gain (8.92-9.35, noise ~0.1) |

Unigram entropy stays the same or rises. At matched corpus diversity, refinement is 10-14% better than lowering the
temperature on TinyStories, and 15-22% better with a hotter draft (`d1.1+r6`: -12% gen-PPL while *more* diverse than
the released sampler on every measure). Ablations: regenerating without re-encoding gives nothing; re-quantizing the
latent instead of the text gives about half. Costs: RT6 uses 17 generator passes instead of 11, 1.9x the wall-clock of the released sampler on an RTX 4090
(the fine levels are the long passes). How non-canonical a draft is (share of its coarse codes the re-encode
changes) predicts its quality on TinyStories (Spearman 0.37 with Gen. PPL; weak on LM1B, where every draft is
rewritten 85-97%); selecting drafts by that signal only trades diversity for perplexity. Details, controls and negative results are in the paper (`paper/main.pdf`).

## Use

The scripts run inside a checkout of the official NCM repository, which provides the model and tokenizer code:

```bash
git clone https://github.com/HazyResearch/ncm.git && cd ncm
git clone https://github.com/PierreKhoury1/ncm-round-trip.git round_trip
pip install -r requirements.txt
hf download hazyresearch/ncm-checkpoints --include "tinystories/*" --local-dir artifacts

# draft + refinements (3 seeds x 256 samples in the paper)
python round_trip/refine_eval.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories \
    --n 256 --seed 1 --variants full,r6,r5x2,r3+r6+r9,t0.8@1 --out round_trip/runs/ts_s1.pkl
python round_trip/analyze_skip.py --in round_trip/runs/ts_s1.pkl --ref gpt2-large
```

Variant names: `full` is the released sampler; `r<k>` one round restarting at level k; `r<k>x<n>` n rounds;
`r4+r8` rounds in sequence; `t<T>@<k>` temperature T for levels >= k; `d<T>` draft temperature; `n<k>` and `q<k>`
are the ablations (no re-encode; latent re-quantization); `b<N>` keeps the most canonical of N drafts. Every
variant also records per-sample non-canonicality (`canon`) and wall-clock (`seconds`, split in `time`). LM1B uses
`score_flm.py` (the authors' FLM protocol), XSum/WMT14 use `--split test` and `score_cond.py`.

## Files

| file | what |
|---|---|
| `refine_eval.py` | round-trip refinement on any released checkpoint, through the authors' `generate` |
| `analyze_skip.py`, `score_flm.py`, `score_cond.py` | scoring: TinyStories, LM1B (FLM protocol), XSum/WMT14 |
| `roundtrip_check.py` | does re-encode + decode reproduce generated text? (CPU, tokenizer only) |
| `judge_pairs.py`, `score_mauve.py` | blind pairwise LLM judge (both orders) and MAUVE against human text |
| `score_ppl.py`, `canon_analyze.py` | any HF causal LM as scorer with per-sample perplexities (second scorer); non-canonicality vs quality |
| `sudoku_search.py` | verifier-guided test-time search on Sudoku (fresh resampling vs round-trip retries) |
| `var_rt.py` | the round trip on the VAR image model (ImageNet 256) with FID screening |
| `train_dropout.py`, `train_joint.py`, `soup.py` | the training-side experiments (negative results) |
| `early_exit.py`, `skip_levels.py`, `oracle_levels.py`, `analyze.py` | early exit / level skipping study |
| `runs/` | the exact pipelines run on rented GPUs |
| `results/LAB_NOTEBOOK.md` | chronological results log, including everything that did not work |
| `paper/` | LaTeX source; `collect.py` rebuilds `results.json` from run outputs, `make_figs.py` the figures and tables |

## Credits

Built on the NCM code, checkpoints and data released by Kumbong et al. (MIT licence, see `THIRD_PARTY_NOTICES`). `pick_tokens` in
`refine_eval.py` and the FLM scoring in `score_flm.py` are copied from the NCM repository's `scripts/eval_shard.py`
and `scripts/eval_merge.py`.
