---
name: maxatac
description: Use this skill whenever the user wants to work with maxATAC, a Python/CLI toolkit for predicting transcription-factor (TF) binding from ATAC-seq (or pseudo-bulk scATAC-seq) signal and DNA sequence in human cell types using deep neural networks. This includes preparing ATAC-seq signal (maxatac prepare), normalizing or averaging bigwig tracks, predicting TF binding sites (maxatac predict), training new models (maxatac train), calling peaks on prediction tracks (maxatac peaks), benchmarking predictions against ChIP-seq gold standards (maxatac benchmark), computing threshold statistics (maxatac threshold), predicting variant-specific binding (maxatac variants), or downloading reference data (maxatac data). Use this skill to help construct correct `maxatac` CLI commands, choose appropriate flags/defaults, interpret inputs/outputs, plan multi-step workflows, and troubleshoot installation or runtime issues. Trigger on mentions of maxATAC, TF binding prediction from ATAC-seq, or any `maxatac` subcommand.
license: Apache-2.0. See LICENSE for complete terms.
---

# maxATAC: TF Binding Prediction from ATAC-seq

## Overview

maxATAC is a Python package and CLI (`maxatac`) that predicts transcription-factor
binding sites (TFBS) genome-wide from ATAC-seq signal and DNA sequence, using
deep neural networks trained per-TF. It works with bulk ATAC-seq and
pseudobulk scATAC-seq, and makes predictions at 32 bp resolution across the
hg38 reference genome (other genomes require user-supplied reference files).

maxATAC requires three kinds of input:

* DNA sequence in `.2bit` format.
* ATAC-seq signal, processed into a normalized `.bigwig` track (via `maxatac prepare`).
* A trained maxATAC TF model in `.h5` format (via `maxatac data`, or trained with `maxatac train`).

This skill is documentation/advisory: it helps you build correct commands,
understand each subcommand's required/optional flags and outputs, and chain
subcommands into a working pipeline. Actually running `maxatac` requires a
dedicated environment (Python 3.9, TensorFlow 2.14, `bedtools`, `samtools`,
`pigz`, `bedGraphToBigWig`, `wget`, `git`, `graphviz`) and, for real
predictions, ~2 GB+ of reference data and potentially large compute (a whole
genome prediction is a large job; each per-TF output bigwig is ~700 MB).

## Installation

```bash
conda create -n maxatac -c bioconda python=3.9 samtools wget bedtools ucsc-bedgraphtobigwig pigz
conda activate maxatac
pip install maxatac
maxatac -h                 # verify install
maxatac data                # download reference data (~/opt/maxatac/data by default)
```

If training fails on a graphviz error, run `conda install graphviz`.
maxATAC pins `tensorflow==2.14.0`; mismatched TensorFlow versions are a
common source of `train`/`predict` errors.

`maxatac data` clones https://github.com/MiraldiLab/maxATAC_data into
`<output>/maxatac/data` (default output: `~/opt`) and downloads the `.2bit`
genome file(s) for the requested `--genome` build(s) (default `hg38`; also
supports `hg19`, `mm10`, or `all`). Default reference/model paths used by
every other subcommand assume data lives at `~/opt/maxatac/data` — pass
explicit `-cs`/`--sequence`/`--blacklist`/`-cutoff_file` flags to override.

## Quick Start: Typical Workflow

1. **Prepare** raw ATAC-seq into a normalized signal track:
   ```bash
   maxatac prepare -i sample.bam -o ./prepare_out -prefix SAMPLE -dedup
   ```
   Produces `SAMPLE_IS_slop20_RP20M_minmax01.bw` — this is the file to feed into `predict`.

2. **Predict** TF binding genome-wide using a pretrained model:
   ```bash
   maxatac predict -tf CTCF -s ./prepare_out/SAMPLE_IS_slop20_RP20M_minmax01.bw -o ./predict_out
   ```
   Outputs a raw score bigwig plus (unless `-skip_call_peaks`) a thresholded `.bed` of TFBS calls.

3. Optionally **benchmark** predictions against a ChIP-seq gold standard, or
   **call peaks** separately with a custom cutoff, or run **variants** to
   test allele-specific effects on binding.

For training a new TF model from your own ATAC-seq + ChIP-seq data, see the
`train` workflow in REFERENCE.md — it requires a tab-delimited meta file
describing cell lines, TFs, and signal/peak file paths.

## Subcommand Quick Reference

| Subcommand | Purpose | Required inputs | Key output |
|---|---|---|---|
| `data` | Download hg38/hg19/mm10 reference data + models | — | `~/opt/maxatac/data/...` |
| `prepare` | BAM/scATAC fragments → normalized signal bigwig | `-i` bam/tsv, `-o`, `-prefix` | `*_minmax01.bw` |
| `average` | Average multiple bigwigs into one | `-i` bigwigs, `-n` name | `<name>.bw` |
| `normalize` | Normalize a bigwig (min-max / zscore / arcsinh) | `-i` signal, `-n` name | normalized `.bw` |
| `predict` | Predict TF binding genome-wide or in ROI | `-tf` or `-m`, `-s` signal, `-n` name | prediction `.bw` (+ `.bed` peaks) |
| `train` | Train a new TF model | `--sequence`, `--meta_file` | `.h5` model, training plots |
| `threshold` | Compute precision/recall/F1 threshold stats for a model | `--prefix`, `--meta_file` | threshold `.tsv` |
| `peaks` | Threshold a prediction bigwig into TFBS `.bed` | `-i` bigwig, `-cutoff_file` | `.bed` peaks |
| `benchmark` | AUPRC of predictions vs. ChIP-seq gold standard | `--prediction`, `--gold_standard`, `--prefix` | stats + PR curve |
| `variants` | Predict allele-specific TF binding at variant positions | `-m` model, `-s` signal, `--variants_bed`, `-n` | prediction `.bw`/`.bed` |

Every subcommand accepts `--loglevel` (fatal/error/warning/info/debug,
default `info`) and most accept `-c`/`--chromosomes` to restrict scope
(default: autosomal `chr1`-`chr22`).

See **REFERENCE.md** for the complete flag list, defaults, and multiple
examples per subcommand — read it before constructing a non-trivial command
or when a flag's exact name/default is needed.

## Common Pitfalls

* `predict`'s `-tf`/`--model` are mutually exclusive — use `-tf CTCF` to
  auto-select the best model + cutoff file, or `-m path.h5` for a specific model.
* `prepare`'s `-slop 20` and `-rpm 20000000` defaults match how the released
  models were trained; changing `-slop` breaks compatibility with pretrained
  models (retraining would be required).
* Models were trained on hg38; predictions on chrX/chrY are unsupported by
  the released models, and other species/genome builds need user-supplied
  `.2bit`, chrom-sizes, and blacklist files.
* A full-genome `predict` run produces a ~700 MB bigwig per TF — for
  exploration, scope with `--chromosomes` or `--roi <regions.bed>` first.
* `train` requires TensorFlow 2.14.0 specifically; version drift is the most
  common cause of training/prediction failures.
