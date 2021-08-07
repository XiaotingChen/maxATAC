# maxATAC: a suite of user-friendly, deep neural network models for transcription factor binding prediction from ATAC-seq

## Introduction

Cellular behavior is the result of complex genomic regulation partially controlled by the activity of DNA binding proteins called transcription factors (TFs). TFs bind DNA in a sequence specific manner to regulate gene transcription. TFs, other DNA binding proteins, nucleosomes, and structural proteins are all involved in the regulation of gene expression. The physical interaction of protein with DNA molecules results in changes in the accessibility of the underlying DNA sequence. The assay for transposase accessible chromatin (ATAC-seq) uses a hyperactive Tn5 transposase to probe for genomic regions that are accessible to cleavage, and in turn, accessible to TF binding. It has been shown that distinct patterns in genomic Tn5 cleavage signal can be used to identify TF binding positions that are partially protected from Tn5 cleavage, known as TF footprints. ATAC-seq can also be used to identify regions of the genome that are generally accessible. Here we present a method to predict TF binding by learning from ATAC-seq accessibility signal and the underlying DNA sequence of TF binding locations identified by ChIP-seq. 


The maxATAC package is a collection of tools used for learning to predict TF binding from ATAC-seq and ChIP-seq data. MaxATAC also provides functions for interpreting trained models and preparing the input data.

## Requirements

This version requires python 3.6 and BEDTools. 

## Installation

It is best to install maxATAC into a dedicated virtual environment. Clone the repository with `git clone` into your `repo` directory of choice. 

Change into the maxATAC repository with `cd maxATAC` and use `pip3 install -e .` to install the package. 

You will also need to have BEDtools installed or loaded on your PATH.

*Note: sometimes SHAP will produce an error when installing with `pip3 install -e .` due to conflicts with numpy. In this case, you will need to install numpy into your virtual env BEFORE installing maxATAC*

## maxATAC Workflow Overview

Steps in training and assessing a maxATAC model. Relevant functions are listed below each step. 

### 1. Prepare Input Data
   * [`average`](./docs/average.md#Average)
   * [`normalize`](./docs/normalize.md#Normalize)
   
### 2. Train a model
   * [`train`](./docs/train.md#Train)
    
### 3. Predict in new cell type
   * `predict`
   
### 4. Benchmark models against experimental data
   * `benchmark`
    
### 5. Learn features import to predict TF binding with a neural network
   * `interpret`

## Walkthroughs

### Predict

The `predict` function takes as input BED formatted genomic regions to predict TF binding using a trained maxATAC model.

BED file requirements for prediction. You must have at least a 3 column file with chromosome, start, and stop coordinates. The interval distance has to be the same as the distance used to train the model. If you trained a model with a resolution 1024, you need to make sure your intervals are spaced 1024 bp apart for prediction with your model.

Example input BED file for prediction:

`chr1   1000    2024`

**Workflow Overview**

1) Create directories and set up filenames
2) Make predictions
3) Convert predictions to bigwig format and write results

Example command:

```bash
maxatac predict --models CTCF_epoch10.h5 --sequence hg38.2bit --signal GM12878__CTCF_slop20bp_RP20M_logp1_minmax01.bw --roi chr1_w1024_PC.bed --prefix test_preds
```

### Benchmark

The `benchmark` function takes as input a prediction bigwig signal track and a ChIP-seq gold standard bigwig track to calculate precision and recall.

The inputs need to be in bigwig format to use this function. You can also provide a custom blacklist to filter out regions that you do not want to include in your comparison. We use a np.mask to exclude these regions.

Currently, benchmarking is set up for one chromosome at a time. The most time-consuming step is importing and binning the input bigwig files to resolutions smaller than 100bp. We are also only benchmarking on whole chromosomes at the moment so everything not in the blacklist will be considered a potential region.

**Workflow Overview**

1) Create directories and set up filenames
2) Get the blacklist mask using the input blacklist and bin it at the same resolution as the predictions and GS
3) Calculate the AUPR

Example command:

```bash
maxatac benchmark --prediction ELK1_slop20_RR30_epoch20_GM12878.bw --gold_standard GM12878__ELK1.bw --prefix ELK1_GM12878_chr1 --output /benchmark_result --bin_size 10000 --chromosomes chr1
```
