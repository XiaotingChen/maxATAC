![maxATAC_logo](https://user-images.githubusercontent.com/47329147/137503708-86d000ef-d6d4-4f75-99aa-39f8aab6dec5.png)

# maxATAC: genome-scale transcription-factor binding prediction from ATAC-seq with deep neural networks

## Introduction

maxATAC is a Python package for transcription factor (TF) binding prediction from ATAC-seq signal and DNA sequence in *human* cell types. maxATAC works with both population-level (bulk) ATAC-seq and pseudobulk ATAC-seq profiles derived from single-cell (sc)ATAC-seq. maxATAC makes TF binding site (TFBS) predictions at 32 bp resolution.
maxATAC requires three inputs:

* DNA sequence, in [`.2bit`](https://genome.ucsc.edu/goldenPath/help/twoBit.html) file format.
* ATAC-seq signal, processed as described [below](#Preparing-your-ATAC-seq-signal).
* Trained maxATAC TF Models, in [`.h5`](https://www.tensorflow.org/tutorials/keras/save_and_load) file format.

>**maxATAC was trained and evaluated on data generated using the hg38 reference genome. The defualt paths and files that are used for each function will reference hg38 files. If you want to use maxATAC with any other species or reference, you will need to provide the appropriate chromosome sizes file, blacklist, and `.2bit` file specific to your data.**

___

## Installation

It is best to install maxATAC into a dedicated virtual environment.

This version requires python 3.9, `bedtools`, `samtools`, `pigz`, `wget`, `git`, and `bedGraphToBigWig` in order to run all functions.

> The total install requirements for maxATAC with reference data are ~2 GB. 

### Installing with Conda

1. Create a conda environment for maxATAC with `conda create -n maxatac python=3.9 maxatac samtools wget bedtools ucsc-bedgraphtobigwig pigz`

2. Test installation with `maxatac -h`

### Installing with pip

1. Create a virtual environment for maxATAC (conda is shown in the example) with `conda create -n maxatac python=3.9`.

2. Install required packages and make sure they are on your PATH: samtools, bedtools, bedGraphToBigWig, wget, git, pigz.

3. Install maxatac with `pip install maxatac`

4. Test installation with `maxatac -h`

### Downloading required reference data

In order to run the maxATAC models that were described in [Cazares et al.](https://www.biorxiv.org/content/10.1101/2022.01.28.478235v1) the following files are required to be downloaded from the [maxATAC_data](https://github.com/MiraldiLab/maxATAC_data) repository, then installed in the correct path:

* hg38 reference genome `.2bit` file
* hg38 chromosome sizes file
* maxATAC extended blacklist
* TF specific `.h5` model files
* TF specific thresholding files
* Bash scripts for preparing data
  
If you do not want to set each specific flag for the above files when running, clone the repository into your `~/opt/` directory under `~/opt/maxatac` using the command:

1. `mkdir -p ~/opt/maxatac && cd ~/opt/maxatac`
2. `git clone https://github.com/MiraldiLab/maxATAC_data.git && mv maxATAC_data data`.
3. `cd ./data/hg38 && wget https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/hg38.2bit`

The directory `~/opt/maxatac/data/hg38` is the default location that maxATAC will look for the data.

The alternative option is to use the command `maxatac data` to download the data to the required directory. Only the hg38 reference genome is supported.

*We are still working on the best way to distribute the data with the package.*
___

## maxATAC Quick Start Overview

![maxATAC Predict Overview](./docs/readme/maxatac_predict_overview.svg)

Schematic: maxATAC prediction of CTCF bindings sites for processed GM12878 ATAC-seq signal

### Inputs

* DNA sequence, in [`.2bit`](https://genome.ucsc.edu/goldenPath/help/twoBit.html) file format.
* ATAC-seq signal, processed as described [below](#preparing-the-atac-seq-signal).
* Trained maxATAC TF Models, in [`.h5`](https://www.tensorflow.org/tutorials/keras/save_and_load) file format.

### Outputs

* Raw maxATAC TFBS scores tracks in [`.bw`](https://genome.ucsc.edu/FAQ/FAQformat.html#format6.1) file format.
* [`.bed`](https://genome.ucsc.edu/FAQ/FAQformat.html#format1) file of TF binding sites, thresholded according to a user-supplied confidence cut off (e.g., corresponding to an estimated precision, recall value or $log_2(precision:precision_{random} > 7$) or default ($max(F1score)$)).

## ATAC-seq Data Requirements

As described in [Cazares et al.](https://www.biorxiv.org/content/10.1101/2022.01.28.478235v1), **maxATAC processing of ATAC-seq signal is critical to maxATAC prediction**. Key maxATAC processing steps, summarized in a single command [`maxatac prepare`](./docs/readme/prepare.md#Prepare), include identification of Tn5 cut sites from ATAC-seq fragments, ATAC-seq signal smoothing, filtering with an extended "maxATAC" blacklist, and robust, min-max-like normalization. 

The maxATAC models were trained on paired-end ATAC-seq data in human. For this reason, we recommend paired-end sequencing with sufficient sequencing depth (e.g., ~20M reads for bulk ATAC-seq). Until these models are benchmarked in other species, we recommend limiting their use to human ATAC-seq datasets. **All the data used to train maxATAC was aligned to the hg38 genome, therefore these models cannot be used on data aligned to the hg19 reference genome.**

### Preparing the ATAC-seq signal

The current `maxatac predict` function requires a normalized ATAC-seq signal in a bigwig format. Use `maxatac prepare` to generate a normalized signal track from a `.bam` file of aligned reads.

#### Bulk ATAC-seq

The function `maxatac prepare` was designed to take an input BAM file that has aligned to the hg38 reference genome. The inputs to `maxatac prepare` are the input bam file, the output directory, and the filename prefix.

```bash
maxatac prepare -i SRX2717911.bam -o ./output -prefix SRX2717911 -dedup
```

This function took 38 minutes for a sample with 52,657,164 reads in the BAM file. This was tested on a 2019 Macbook Pro with a 2.6 GHz 6-Core Intel Core i7 and 16 GB of memory.

#### Pseudo-bulk scATAC-seq

First, convert the `.tsv.gz` output fragments file from CellRanger into pseudo-bulk specific fragment files. Then, use `maxatac prepare` with each of the fragment files in order to generate a normalized bigwig file for input into `maxatac predict`.

```bash
maxatac prepare -i HighLoading_GM12878.tsv -o ./output -prefix HighLoading_GM12878
```

The prediction parameters and steps are the same for scATAC-seq data after normalization.

## Predicting TF binding from ATAC-seq

Following maxATAC-specific processing of ATAC-seq signal inputs, use the [`maxatac predict`](./docs/readme/predict.md#Predict) function to predict TF binding with a maxATAC model.

TF binding predictions can be made genome-wide, for a single chromosome, or, alternatively, the user can provide a `.bed` file of genomic intervals for maxATAC predictions to be made.

The trained maxATAC models, reference `.2bit`, `chrom.sizes`, and maxATAC blacklist files should be downloaded and installed automatically with [Installation](#Installation).

### Whole genome prediction

Example command for TFBS prediction across the whole genome:

```bash
maxatac predict --sequence hg38.2bit --models CTCF.h5 --signal GM12878.bigwig
```

### Prediction in a specific genomic region(s)

For TFBS predictions within specific regions of the genome, a `BED` file of genomic intervals, `roi` (regions of interest) are supplied:

```bash
maxatac predict --sequence hg38.2bit --models CTCF.h5 --signal GM12878.bigwig --roi ROI.bed
```

### Prediction on a specific chromosome(s)

For TFBS predictions on a single chromosome or subset of chromosomes, these can be provided using the `--chromosomes` argument:

```bash
maxatac predict --sequence hg38.2bit --models CTCF.h5 --signal GM12878.bigwig --chromosomes chr3 chr5
```

## Raw signal tracks are large
Each output prediction file for a whole genome is ~700 MB per TF. 

The output bed files are ~60Mb.

There are 127 TF models x ~700MB per TF model = ~88.9 GB of bigwig files for a single ATAC-seq input track.

___

## maxATAC functions

| Subcommand                                          | Description                                    |
|-----------------------------------------------------|------------------------------------------------|
| [`prepare`](./docs/readme/prepare.md#Prepare)       | Prepare input data                             |
| [`average`](./docs/readme/average.md#Average)       | Average ATAC-seq signal tracks                 |
| [`normalize`](./docs/readme/normalize.md#Normalize) | Minmax normalize ATAC-seq signal tracks        |
| [`train`](./docs/readme/train.md#Train)             | Train a model                                  |
| [`predict`](./docs/readme/predict.md#Predict)       | Predict TF binding                             |
| [`benchmark`](./docs/readme/benchmark.md#Benchmark) | Benchmark maxATAC predictions against ChIP-seq |
| [`peaks`](./docs/readme/peaks.md#Peaks)             | Call "peaks" on maxATAC signal tracks          |
| [`variants`](./docs/readme/variants.md#Variants)    | Predict sequence specific TF binding           |

___

## Publication

The maxATAC pre-print is currently available on [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.01.28.478235v1.article-metrics). 

```pre
maxATAC: genome-scale transcription-factor binding prediction from ATAC-seq with deep neural networks
Tareian Cazares, Faiz W. Rizvi, Balaji Iyer, Xiaoting Chen, Michael Kotliar, Joseph A. Wayman, Anthony Bejjani, Omer Donmez, Benjamin Wronowski, Sreeja Parameswaran, Leah C. Kottyan, Artem Barski, Matthew T. Weirauch, VB Surya Prasath, Emily R. Miraldi
bioRxiv 2022.01.28.478235; doi: https://doi.org/10.1101/2022.01.28.478235
```
[Code to generate most of our figures](https://github.com/MiraldiLab/maxATAC_docs/tree/main/figure_code)
