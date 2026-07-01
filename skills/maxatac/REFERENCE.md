# maxATAC CLI Reference

Full flag reference for every `maxatac` subcommand. All subcommands share a
top-level `--genome` flag (default `hg38`) and, where applicable, a
`--loglevel` flag (`fatal`/`error`/`warning`/`info`/`debug`, default `info`).

Default reference files (chrom sizes, blacklist, `.2bit` sequence, per-TF
models and cutoff files) are resolved relative to `~/opt/maxatac/data`,
populated by `maxatac data`. Pass the corresponding flag explicitly to use
custom files/genomes.

---

## `data` — Download reference data

Downloads the [maxATAC_data](https://github.com/MiraldiLab/maxATAC_data) repo
(models, blacklist, chrom sizes, prep scripts) plus the requested genome
`.2bit` file(s) into `<output>/maxatac/data`.

```bash
maxatac data
maxatac data --genome hg38 hg19 --output ~/opt
```

| Flag | Default | Description |
|---|---|---|
| `--genome` | `hg38` | One or more genome builds to download (`hg38`, `hg19`, `mm10`, or `all`). |
| `-o`, `--output` | `~/opt` | Destination directory; final data lands in `<output>/maxatac/data`. |
| `--loglevel` | `info` | Logging level. |

---

## `prepare` — Convert raw ATAC-seq to normalized signal

Converts a BAM (bulk ATAC-seq) or 10x scATAC fragments `.tsv`/`.tsv.gz` file
into Tn5 cut sites, smooths them, read-depth normalizes, and min-max
normalizes into a prediction-ready bigwig. Requires `samtools`, `bedtools`,
`pigz`, and `bedGraphToBigWig` on `PATH`.

```bash
# Bulk ATAC-seq
maxatac prepare -i SRX2717911.bam -o ./output -prefix SRX2717911 -dedup

# Pseudo-bulk scATAC-seq
maxatac prepare -i HighLoading_GM12878.tsv -o ./output -prefix HighLoading_GM12878
```

### Required
| Flag | Description |
|---|---|
| `-i`, `--input` | Input `.bam` (bulk) or `.tsv`/`.tsv.gz` (10x scATAC fragments). |
| `-o`, `--output` | Output directory. |
| `-n`, `-name`, `--prefix`, `-prefix` | Filename prefix for all outputs. |

### Optional
| Flag | Default | Description |
|---|---|---|
| `-skip_dedup`, `--skip_deduplication` | off (dedup runs) | Skip PCR-duplicate removal. |
| `-slop`, `--slop` | `20` | bp to extend each Tn5 cut site. **Do not change** when using released pretrained models. |
| `-rpm`, `--rpm_factor` | `20000000` | Read-depth normalization scaling factor (RP20M). |
| `--blacklist` | hg38 maxATAC blacklist `.bed` | Regions to exclude. |
| `--blacklist_bw` | hg38 maxATAC blacklist `.bw` | Blacklist as bigwig. |
| `-cs`, `--chrom_sizes`, `--chromosome_sizes` | hg38 chrom sizes | Chromosome sizes file. |
| `-c`, `-chroms`, `--chromosomes` | autosomal chr1-22 | Chromosomes to output. |
| `-t`, `-threads`, `--threads` | available CPU count | Threads to use. |
| `--loglevel` | `info` | Logging level. |

### Outputs (prefix `GM12878_scatac_1M`)
| Filename suffix | Description |
|---|---|
| `_IS_slop20.bed.gz` | Compressed cut-site bed, Tn5-shift corrected. |
| `_IS_slop20_RP20M.bw` | Read-depth-normalized signal. |
| `_IS_slop20_RP20M_minmax01.bw` | **Min-max normalized — this is the file to pass to `predict`.** |
| `_IS_slop20_RP20M_minmax01_chromosome_min_max.txt` | Per-chromosome min/max values. |
| `_IS_slop20_RP20M_minmax01_genome_stats.txt` | Genome-wide min/max/median stats. |

---

## `average` — Average multiple bigwigs

```bash
maxatac average -i *.bw -n IMR-90 -o ./test -c chr1 -cs hg38.chrom.sizes
```

### Required
| Flag | Description |
|---|---|
| `-i` | Input bigwig files (glob or list). |
| `-n`, `--name`, `--prefix` | Output filename base (`.bw` appended). |

### Optional
| Flag | Default | Description |
|---|---|---|
| `-cs`, `--chrom_sizes`, `--chromosome_sizes` | hg38 chrom sizes | Chrom sizes file. |
| `-c`, `--chroms`, `--chromosomes` | autosomal chr1-22 | Chromosomes to average/write. |
| `-o`, `--output`, `--output_dir` | cwd | Output directory. |
| `--loglevel` | `info` | Logging level. |

---

## `normalize` — Normalize a bigwig signal track

Methods: `min-max` (default; scale to `[0,1]`, optionally clipped at a
percentile max — 99th percentile by default, more robust to outliers than
absolute max), `zscore` (mean 0, sd 1), `arcsinh` (inverse hyperbolic sine).

```bash
maxatac normalize -i GM12878_RP20M.bw -name GM12878_minmax -o ./test --method min-max --max_percentile 99
```

### Required
| Flag | Description |
|---|---|
| `-i`, `--signal` | Input bigwig. |
| `-n`, `--name`, `--prefix` | Output filename base. |

### Optional
| Flag | Default | Description |
|---|---|---|
| `--method` | `min-max` | `min-max`, `zscore`, or `arcsinh`. |
| `--max_percentile` | `99` | Percentile used as max for `min-max`. |
| `--min` | `0` | Min value for `min-max`. |
| `--max` | computed from data | Max value for `min-max`. |
| `--clip` | `False` | Clip values above max instead of leaving as-is. |
| `-c`, `--chroms`, `--chromosomes` | autosomal chr1-22 | Chromosomes to normalize. |
| `-cs`, `--chrom_sizes`, `--chromosome_sizes` | hg38 chrom sizes | Chrom sizes file. |
| `--blacklist_bw` | hg38 maxATAC blacklist | Regions to exclude (bigwig). |
| `-o`, `--output`, `--output_dir` | `./normalize` | Output directory. |
| `--loglevel` | `info` | Logging level. |

---

## `predict` — Predict TF binding

```bash
maxatac predict -tf CTCF --signal GM12878_IS_slop20_RP20M_minmax01.bw -o outputdir/
maxatac predict -tf CTCF --signal GM12878_IS_slop20_RP20M_minmax01.bw --roi ROI.bed
maxatac predict -tf CTCF --signal GM12878_IS_slop20_RP20M_minmax01.bw --chromosomes chr3 chr5
```

### Required
| Flag | Description |
|---|---|
| `-tf`, `--tf_name` **or** `-m`, `--model` | Mutually exclusive. `-tf` auto-selects the best model + cutoff file; `-m` points at a specific `.h5` model. |
| `-i`, `-s`, `--signal` | Input normalized ATAC-seq bigwig (from `prepare`). |
| `-n`, `--name`, `--prefix` | Output filename prefix. |

### Optional
| Flag | Default | Description |
|---|---|---|
| `--seq`, `--sequence` | hg38 `.2bit` | Genome sequence file. |
| `-o`, `--output` | `./prediction_results` | Output directory. |
| `-bl`, `--blacklist` | maxATAC blacklist | Regions to exclude. |
| `--bed`, `--peaks`, `--regions`, `--roi`, `-roi` | none (whole chromosome) | BED of regions to restrict/refine prediction windows. |
| `--windows`, `-w` | none | Precomputed 1,024 bp windows (fixed step) to use instead of generating de novo. |
| `--batch_size` | `10000` | Regions per prediction batch; lower if memory-constrained. |
| `--step_size` | `INPUT_LENGTH/4` = `256` | Step size for sliding prediction windows (overlaps averaged). |
| `-cs`, `--chrom_sizes`, `--chromosome_sizes` | hg38 chrom sizes | Chrom sizes file. |
| `-c`, `--chromosomes` | autosomal chr1-22 | Chromosomes to predict (chrX/chrY unsupported by released models). |
| `-ct`, `--cutoff_type` | `F1` | `Precision`, `Recall`, `F1`, or `log2FC` (log2(precision:random precision)). |
| `-cv`, `--cutoff_value` | none | Cutoff value for `cutoff_type` (not used with `F1`). |
| `-cf`, `--cutoff_file` | auto (from `-tf`) | Path to TF-specific threshold stats file (`/data/models`). |
| `-skip_call_peaks`, `--skip_call_peaks` | `False` | Skip peak-calling step at the end. |
| `--threads` | `24` | Parallel processes (set to # GPUs if using GPUs). |
| `--loglevel` | `info` | Logging level. |

---

## `train` — Train a new TF model

Requires a tab-delimited **meta file** with columns:

| Column | Description |
|---|---|
| `Cell_Line` | Sample cell type |
| `TF` | Gene symbol for the TF |
| `ATAC_Signal_File` | Path to ATAC-seq bigwig |
| `Binding_File` | Path to ChIP-seq bigwig |
| `ATAC_Peaks` | Path to ATAC-seq peak BED |
| `CHIP_Peaks` | Path to ChIP-seq peak BED |
| `Train_Test_Label` | `Train` or `Test` |

Note: maxATAC pins TensorFlow 2.14.0 — mismatched TF versions are the most
common cause of `train` errors.

```bash
maxatac train --arch DCNN_V2 --sequence hg38.2bit --meta_file CTCF_meta.tsv \
  --output ./CTCF_DCNN --prefix CTCF_DCNN --shuffle_cell_type --rev_comp
```

### Required
| Flag | Description |
|---|---|
| `--genome` | Genome build (e.g. `hg38`). |
| `--sequence` | `.2bit` DNA sequence file. |
| `--meta_file` | Path to the meta file described above. |

### Optional
| Flag | Default | Description |
|---|---|---|
| `--prefix` | `maxatac_model` | Output filename prefix. |
| `--train_roi` | none (built from meta peaks) | BED of training ROIs, overrides peak-derived ROI pool. |
| `--validate_roi` | none | BED of validation ROIs, overrides peak-derived pool. |
| `--target_scale_factor` | `1` | Target-signal scaling factor (quantitative models only). |
| `--output_activation` | `sigmoid` | Output layer activation. |
| `--chroms` | `chr2-7,9-22,X` (excludes 1,8) | Chromosomes considered for train+validate. |
| `--tchroms` | subset excluding 1,2,8,19,X,Y,M | Training-only chromosomes. |
| `--vchroms` | `chr2 chr19` | Validation-only chromosomes. |
| `--arch` | `DCNN_V2` | Architecture: `DCNN_V2`, `RES_DCNN_V2`, `MM_DCNN_V2`, `MM_Res_DCNN_V2`. |
| `--rand_ratio` | `0` | Fraction of each batch drawn from random (non-peak) genomic regions. |
| `--seed` | random `[1, 99999]` | RNG seed. |
| `--weights` | none | `.h5` weights to initialize from. |
| `--epochs` | `20` | Training epochs. |
| `--batches` | `100` | Batches per epoch. |
| `--batch_size` | `1000` | Examples per training batch. |
| `--val_batch_size` | `1000` | Examples per validation batch. |
| `--output` | `./training_results` | Output directory. |
| `--plot` | `True` | Plot model structure + training history. |
| `--dense` | `False` | Add a dense layer before model output. |
| `--threads` | available CPU count | Parallel threads. |
| `--shuffle_cell_type` | `True` | Shuffle training-ROI cell-type label ("pan-cell" training). |
| `--rev_comp` | `False` | Also train on reverse-complement sequence. |
| `--multiprocessing` | `False` | Use multiprocessing with `tf.keras.fit()`. |
| `--max_queue_size` | none | Max data-loading worker queue size. |
| `--save_roi` | `False` | Save ROI files/stats generated for training. |
| `--blacklist` | maxATAC blacklist | Regions to exclude. |
| `--chrom_sizes` | hg38 chrom sizes | Chrom sizes file. |
| `--loglevel` | `info` | Logging level. |

Training approach: builds train/validate/test ROI pools from pooled
ATAC+ChIP peaks per TF; each epoch draws 100 batches of 1,000
"peak-centric", "pan-cell" examples; one cell type + 2 chromosomes are held
out per TF for independent testing; best model chosen by dice coefficient.

---

## `threshold` — Compute model threshold statistics

Generates precision/recall/F1/log2FC threshold statistics for a trained
model against a gold standard, for use as a `-cutoff_file` in `predict`/`peaks`.

| Flag | Default | Description |
|---|---|---|
| `--prefix` | *(required)* | Output filename prefix. |
| `--meta_file` | *(required)* | Meta file with prediction-signal and gold-standard paths per cell line (`.tsv`). |
| `--chrom_sizes` | hg38 chrom sizes | Chrom sizes file. |
| `--chromosomes` | `chr2 chr19` (default validate chroms) | Chromosomes to use. |
| `--bin_size` | `200` | Aggregation bin size. |
| `--output` | `./threshold` | Output directory. |
| `--blacklist_bw` | maxATAC blacklist | Regions to exclude. |
| `--loglevel` | `info` | Logging level. |

---

## `peaks` — Call TFBS peaks from a prediction bigwig

```bash
maxatac peaks -i GM12878_CTCF.bw -o ./peaks -bin 32 -cutoff_file ARID3A_validationPerformance_vs_thresholdCalibration.tsv
```

### Required
| Flag | Description |
|---|---|
| `-i`, `--input_bigwig` | Input maxATAC prediction bigwig. |
| `-cutoff_file`, `--cutoff_file` | TF-specific threshold stats file (`/data/models`). |

### Optional
| Flag | Default | Description |
|---|---|---|
| `-cutoff_type`, `--cutoff_type` | `F1` | `Precision`, `Recall`, `F1`, or `log2FC`. |
| `-cutoff_value`, `--cutoff_value` | none | Value for `cutoff_type` (e.g. `.7`; precision/recall/F1 are 0-1, log2FC is 0-inf). |
| `-prefix`, `--prefix` | strip `.bw` from input | Output filename prefix. |
| `-bin`, `--bin_size` | `32` | Bin size (bp) for TFBS intervals. |
| `-o`, `--output` | `./peaks` | Output directory. |
| `--chromosomes` | autosomal chr1-22 | Chromosomes to limit peak calling to. |
| `--loglevel` | `info` | Logging level. |

---

## `benchmark` — Score predictions against a gold standard

Computes AUPRC comparing a prediction track (bed or bigwig) to a binary
ChIP-seq-derived gold-standard bigwig (1 = TFBS, 0 = no TFBS).

```bash
maxatac benchmark --prediction GM12878_CTCF_chr1.bw --gold_standard GM12878_CTCF_ENCODE_IDR.bw --chromosomes chr1 --bin_size 200
```

### Required
| Flag | Description |
|---|---|
| `-bed`, `--bed` **or** `--bw`, `--bigwig`, `-bw` | Prediction file (mutually exclusive: BED or bigwig). |
| `--gold_standard` | Binary gold-standard bigwig. |
| `-n`, `--name`, `--prefix` | Output filename prefix. |

### Optional
| Flag | Default | Description |
|---|---|---|
| `-c`, `--chroms`, `--chromosomes` | `chr1 chr8` (held-out test chroms) | Chromosomes to benchmark. |
| `--bin_size` | `200` | Bin size for aggregating base-pair predictions (ENCODE-DREAM challenge convention). |
| `--agg` | `max` | Aggregation function: `max`, `min`, `mean`. |
| `--round_predictions` | `9` | Decimal precision for predictions (TensorFlow's practical floor). |
| `-o`, `--output_directory` | `./benchmarking_results` | Output directory. |
| `--blacklist_bw` | hg38 maxATAC blacklist | Regions to exclude. |
| `-skip_plot`, `--skip_plot` | `False` | Skip PR-curve plotting. |
| `--loglevel` | `info` | Logging level. |

---

## `variants` — Predict allele-specific TF binding

Predicts TF binding while substituting specific nucleotides (e.g. at
variant positions), for non-overlapping LD blocks. Whole-genome prediction
is possible but not yet optimized for this subcommand.

```bash
maxatac variants -m ELF1_99.h5 -signal GM12878__slop20bp_RP20M_minmax01.bw \
  -name GM12878_ELF1 -s hg38.2bit --chromosomes chr20 -variants_bed AD_risk_loci.bed
```

### Required
| Flag | Description |
|---|---|
| `--genome` | Genome build. |
| `-m`, `--model` | Trained `.h5` model. |
| `-i`, `-s`, `--signal` | ATAC-seq signal bigwig. |
| `-variants_bed`, `--variants_bed` | BED of variant positions; first 3 columns = coordinates, 4th column = nucleotide to substitute. |
| `-n`, `--name`, `--prefix` | Output filename prefix. |

### Optional
| Flag | Default | Description |
|---|---|---|
| `-s`, `--sequence` | hg38 `.2bit` | DNA sequence file. |
| `-roi`, `--roi` | whole genome | BED of intervals (LD blocks) to restrict prediction to; only first 3 columns used. |
| `-o`, `--output_dir` | `./variantss` | Output directory (note: upstream typo, double `s`). |
| `-step_size`, `--step_size` | `256` | Overlap step size (bp), must be a multiple of 256. |
| `-c`, `-chroms`, `--chromosomes` | all chr1-22,X,Y | Chromosomes to predict. |
| `-cs`, `--chrom_sizes` | hg38 chrom sizes | Chrom sizes file. |
| `--blacklist` | maxATAC blacklist | Regions to exclude. |
| `--loglevel` | `info` | Logging level. |

Behavior: merges nearby ROI intervals (±512 bp), creates sliding
1,024 bp windows with 256 bp step across each ROI, trims trailing windows
shorter than 1,024 bp.
