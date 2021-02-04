import numpy as np
import pandas as pd

import random

from maxatac.utilities.bigwig import load_bigwig, safe_load_bigwig
from maxatac.utilities.twobit import load_2bit
from maxatac.utilities.constants import (
    INPUT_CHANNELS,
    INPUT_LENGTH,
    BATCH_SIZE,
    VAL_BATCH_SIZE,
    CHR_POOL_SIZE,
    BP_ORDER,
    TRAIN_SCALE_SIGNAL,
    QUANT_TARGET_SCALE_FACTOR
)
from maxatac.utilities.prepare import (
    RandomRegionsPool,
    get_input_matrix
)


def get_roi_pool_predict(seq_len=None, roi=None, shuffle=False, tf=None, cl=None):
    roi_df = pd.read_csv(roi, sep="\t", header=0, index_col=None)
    temp = roi_df['Stop'] - roi_df['Start']
    ##############################

    # Temporary Workaround. Needs to be deleted later
    roi_ok = (temp == seq_len)
    temp_df = roi_df[roi_ok == True]
    roi_df = temp_df
    ###############################

    # roi_ok = (temp == seq_len).all()
    # if not roi_ok:

    # sys.exit("ROI Length Does Not Match Input Length")
    roi_df['TF'] = tf
    roi_df['Cell_Line'] = cl
    if shuffle:
        roi_df = roi_df.sample(frac=1)
    return roi_df


def get_roi_pool(seq_len=None, roi=None, shuffle=False):
    roi_df = pd.read_csv(roi, sep="\t", header=0, index_col=None)
    temp = roi_df['Stop'] - roi_df['Start']
    roi_ok = (temp == seq_len)
    temp_df = roi_df[roi_ok == True]
    roi_df = temp_df

    if shuffle:
        roi_df = roi_df.sample(frac=1)
    return roi_df


def get_one_hot_encoded(sequence, target_bp):
    one_hot_encoded = []
    for s in sequence:
        if s.lower() == target_bp.lower():
            one_hot_encoded.append(1)
        else:
            one_hot_encoded.append(0)
    return one_hot_encoded


def get_pc_input_matrix(
        rows,
        cols,
        batch_size,  # make sure that cols % batch_size == 0
        signal_stream,
        average_stream,
        sequence_stream,
        bp_order,
        chrom,
        start,  # end - start = cols
        end,
        reshape=True,

):
    input_matrix = np.zeros((rows, cols))
    for n, bp in enumerate(bp_order):
        input_matrix[n, :] = get_one_hot_encoded(
            sequence_stream.sequence(chrom, start, end),
            bp
        )

    signal_array = np.array(signal_stream.values(chrom, start, end))
    avg_array = np.array(average_stream.values(chrom, start, end))
    input_matrix[4, :] = signal_array
    input_matrix[5, :] = input_matrix[4, :] - avg_array
    input_matrix = input_matrix.T

    if reshape:
        input_matrix = np.reshape(
            input_matrix,
            (batch_size, round(cols / batch_size), rows)
        )

    return input_matrix


def make_pc_pred_batch(
        batch_idxs,
        sequence,
        average,
        meta_table,
        roi_pool,
        bp_resolution=1,
        filters=None
):
    roi_size = roi_pool.shape[0]
    # batch_idx=0
    # n_batches = int(roi_size/BATCH_SIZE)
    # Here I will process by row, if performance is bad then process by cell line

    with \
            safe_load_bigwig(filters) as filters_stream, \
            load_bigwig(average) as average_stream, \
            load_2bit(sequence) as sequence_stream:

        inputs_batch, targets_batch = [], []
        batch_meta_df = pd.DataFrame()
        batch_gold_vals = []
        for row_idx in batch_idxs:
            roi_row = roi_pool.iloc[row_idx, :]
            cell_line = roi_row['Cell_Line']
            tf = roi_row['TF']
            chrom_name = roi_row['Chr']
            start = int(roi_row['Start'])
            end = int(roi_row['Stop'])
            meta_row = meta_table[((meta_table['Cell_Line'] == cell_line) & (meta_table['TF'] == tf))]
            meta_row = meta_row.reset_index(drop=True)
            meta_row["Start"] = start
            meta_row["Stop"] = end
            meta_row["Chr"] = chrom_name
            try:
                signal = meta_row.loc[0, 'ATAC_Signal_File']
                binding = meta_row.loc[0, 'Binding_File']
            except:
                print("Error in creating input batch")
                print(roi_row)

                # sys.exit("Here=1. Error while creating input batch")

            with \
                    load_bigwig(binding) as binding_stream, \
                    load_bigwig(signal) as signal_stream:
                try:
                    input_matrix = get_pc_input_matrix(rows=INPUT_CHANNELS,
                                                       cols=INPUT_LENGTH,
                                                       batch_size=1,  # we will combine into batch later
                                                       reshape=False,
                                                       bp_order=BP_ORDER,
                                                       signal_stream=signal_stream,
                                                       average_stream=average_stream,
                                                       sequence_stream=sequence_stream,
                                                       chrom=chrom_name,
                                                       start=start,
                                                       end=end
                                                       )
                    inputs_batch.append(input_matrix)
                    batch_meta_df = pd.concat([batch_meta_df, meta_row], axis='index', ignore_index=True)
                    target_vector = np.array(binding_stream.values(chrom_name, start, end)).T
                    target_vector = np.nan_to_num(target_vector, 0.0)
                    n_bins = int(target_vector.shape[0] / bp_resolution)
                    split_targets = np.array(np.split(target_vector, n_bins, axis=0))
                    bin_sums = np.sum(split_targets, axis=1)
                    bin_vector = np.where(bin_sums > 0.5 * bp_resolution, 1.0, 0.0)
                    batch_gold_vals.append(bin_vector)

                except:
                    print("Error in creating input batch")
                    print(roi_row)
                    continue
                    # sys.exit("Error while creating input batch")
        batch_meta_df = batch_meta_df.drop(['ATAC_Signal_File', 'Binding_File'], axis='columns')
        batch_meta_df.reset_index(drop=True)
        return (np.array(inputs_batch), np.array(batch_gold_vals), batch_meta_df)


def create_roi_batch(
        sequence,
        average,
        meta_table,
        roi_pool,
        n_roi,
        train_tf,
        tchroms,
        bp_resolution=1,
        quant=False,
        filters=None
):
    while True:
        inputs_batch, targets_batch = [], []
        roi_size = roi_pool.shape[0]

        curr_batch_idxs = random.sample(range(roi_size), n_roi)

        # Here I will process by row, if performance is bad then process by cell line
        for row_idx in curr_batch_idxs:
            roi_row = roi_pool.iloc[row_idx, :]
            cell_line = roi_row['Cell_Line']
            tf = train_tf
            chrom_name = roi_row['Chr']

            start = int(roi_row['Start'])
            end = int(roi_row['Stop'])
            meta_row = meta_table[((meta_table['Cell_Line'] == cell_line) & (meta_table['TF'] == tf))]
            meta_row = meta_row.reset_index(drop=True)

            signal = meta_row.loc[0, 'ATAC_Signal_File']
            binding = meta_row.loc[0, 'Binding_File']

            with \
                    load_bigwig(average) as average_stream, \
                    load_2bit(sequence) as sequence_stream, \
                    load_bigwig(signal) as signal_stream, \
                    load_bigwig(binding) as binding_stream:
                input_matrix = get_pc_input_matrix(
                    rows=INPUT_CHANNELS,
                    cols=INPUT_LENGTH,
                    batch_size=1,  # we will combine into batch later
                    reshape=False,
                    bp_order=BP_ORDER,
                    signal_stream=signal_stream,
                    average_stream=average_stream,
                    sequence_stream=sequence_stream,
                    chrom=chrom_name,
                    start=start,
                    end=end
                )

                inputs_batch.append(input_matrix)

                if not quant:
                    target_vector = np.array(binding_stream.values(chrom_name, start, end)).T
                    target_vector = np.nan_to_num(target_vector, 0.0)
                    n_bins = int(target_vector.shape[0] / bp_resolution)
                    split_targets = np.array(np.split(target_vector, n_bins, axis=0))
                    bin_sums = np.sum(split_targets, axis=1)
                    bin_vector = np.where(bin_sums > 0.5 * bp_resolution, 1.0, 0.0)
                    targets_batch.append(bin_vector)

                else:
                    target_vector = np.array(binding_stream.values(chrom_name, start, end)).T
                    target_vector = np.nan_to_num(target_vector, 0.0)
                    n_bins = int(target_vector.shape[0] / bp_resolution)
                    split_targets = np.array(np.split(target_vector, n_bins, axis=0))
                    bin_vector = np.mean(split_targets,
                                         axis=1)  # Perhaps we can change np.mean to np.median. Something to think about.
                    targets_batch.append(bin_vector)

        if quant:
            targets_batch = np.array(targets_batch)
            targets_batch = targets_batch * QUANT_TARGET_SCALE_FACTOR

        yield (np.array(inputs_batch), np.array(targets_batch))


def create_random_batch(
        sequence,
        average,
        meta_table,
        train_cell_lines,
        n_rand,
        train_tf,
        regions_pool,
        bp_resolution=1,
        quant=False,
        filters=None
):
    while True:
        inputs_batch, targets_batch = [], []
        for idx in range(n_rand):
            cell_line = random.choice(train_cell_lines)  # Randomly select a cell line
            chrom_name, seq_start, seq_end = regions_pool.get_region()  # returns random region (chrom_name, start, end) 
            meta_row = meta_table[((meta_table['Cell_Line'] == cell_line) & (
                    meta_table['TF'] == train_tf))]  # get meta table row corresponding to selected cell line
            meta_row = meta_row.reset_index(drop=True)
            signal = meta_row.loc[0, 'ATAC_Signal_File']
            binding = meta_row.loc[0, 'Binding_File']
            with \
                    safe_load_bigwig(filters) as filters_stream, \
                    load_bigwig(average) as average_stream, \
                    load_2bit(sequence) as sequence_stream, \
                    load_bigwig(signal) as signal_stream, \
                    load_bigwig(binding) as binding_stream:
                try:
                    input_matrix = get_input_matrix(
                        rows=INPUT_CHANNELS,
                        cols=INPUT_LENGTH,
                        batch_size=1,  # we will combine into batch later
                        reshape=False,
                        bp_order=BP_ORDER,
                        signal_stream=signal_stream,
                        average_stream=average_stream,
                        sequence_stream=sequence_stream,
                        chrom=chrom_name,
                        start=seq_start,
                        end=seq_end,
                        scale_signal=TRAIN_SCALE_SIGNAL,
                        filters_stream=filters_stream
                    )
                    inputs_batch.append(input_matrix)
                    if not quant:
                        target_vector = np.array(binding_stream.values(chrom_name, seq_start, seq_end)).T
                        target_vector = np.nan_to_num(target_vector, 0.0)
                        n_bins = int(target_vector.shape[0] / bp_resolution)
                        split_targets = np.array(np.split(target_vector, n_bins, axis=0))
                        bin_sums = np.sum(split_targets, axis=1)
                        bin_vector = np.where(bin_sums > 0.5 * bp_resolution, 1.0, 0.0)
                        targets_batch.append(bin_vector)
                    else:
                        target_vector = np.array(binding_stream.values(chrom_name, seq_start, seq_end)).T
                        target_vector = np.nan_to_num(target_vector, 0.0)
                        n_bins = int(target_vector.shape[0] / bp_resolution)
                        split_targets = np.array(np.split(target_vector, n_bins, axis=0))
                        bin_vector = np.mean(split_targets,
                                             axis=1)  # Perhaps we can change np.mean to np.median. Something to think about.
                        targets_batch.append(bin_vector)


                except:
                    here = 2
                    continue
        if quant:
            targets_batch = np.array(targets_batch)
            targets_batch = targets_batch * QUANT_TARGET_SCALE_FACTOR
        yield (np.array(inputs_batch), np.array(targets_batch))


def train_generator(
        sequence,
        average,
        meta_table,
        roi_pool,
        train_cell_lines,
        rand_ratio,
        train_tf,
        tchroms,
        bp_resolution=1,
        quant=False,
        batch_size=BATCH_SIZE
):
    """
    The training data generator will produce batches of training data using the ROI pool and meta table

    :param sequence: The input 2bit dna sequence file
    :param average: The average ATAC-seq signal input file
    :param meta_table: The meta table that is used to find the labels and positions
    :param roi_pool: The roi pool that is used for selecting training and validation regions
    :param train_cell_lines: The training cell lines to use
    :param rand_ratio: The ratio of random regions to mix with ROI examples
    :param train_tf: The TF that you are training the model for
    :param tchroms: The training chromosomes
    :param bp_resolution: The base pair resolution to output the ChIP-seq signal in
    :param quant: Whether the data is quantitative or binary
    :param batch_size: The batch size of the data

    :return: A generator that will mix ROI and random regions together for training
    """
    n_roi = round(batch_size * (1. - rand_ratio))
    n_rand = round(batch_size - n_roi)
    train_random_regions_pool = RandomRegionsPool(
        chroms=tchroms,
        chrom_pool_size=CHR_POOL_SIZE,
        region_length=INPUT_LENGTH,
        preferences=False  # can be None
    )

    roi_gen = create_roi_batch(sequence,
                               average,
                               meta_table,
                               roi_pool,
                               n_roi,
                               train_tf,
                               tchroms,
                               bp_resolution=bp_resolution,
                               quant=quant,
                               filters=None
                               )

    rand_gen = create_random_batch(sequence,
                                   average,
                                   meta_table,
                                   train_cell_lines,
                                   n_rand,
                                   train_tf,
                                   train_random_regions_pool,
                                   bp_resolution=bp_resolution,
                                   quant=quant,
                                   filters=None
                                   )

    while True:
        if 0. < rand_ratio < 1.:
            roi_input_batch, roi_target_batch = next(roi_gen)
            rand_input_batch, rand_target_batch = next(rand_gen)
            inputs_batch = np.concatenate((roi_input_batch, rand_input_batch), axis=0)
            targets_batch = np.concatenate((roi_target_batch, rand_target_batch), axis=0)

        elif rand_ratio == 1.:
            rand_input_batch, rand_target_batch = next(rand_gen)
            inputs_batch = rand_input_batch
            targets_batch = rand_target_batch

        else:
            roi_input_batch, roi_target_batch = next(roi_gen)
            inputs_batch = roi_input_batch
            targets_batch = roi_target_batch

        yield (inputs_batch, targets_batch)


def create_val_generator(
        sequence,
        average,
        meta_table,
        train_cell_lines,
        train_tf,
        all_val_regions,
        bp_resolution=1,
        quant=False,
        filters=None,
        val_batch_size=VAL_BATCH_SIZE
):
    while True:

        inputs_batch, targets_batch = [], []
        n_val_batches = round(all_val_regions.shape[0] / val_batch_size)
        all_batch_idxs = np.array_split(np.arange(all_val_regions.shape[0]), n_val_batches)

        for idx, batch_idxs in enumerate(all_batch_idxs):
            inputs_batch, targets_batch = [], []
            for row_idx in batch_idxs:
                roi_row = all_val_regions.iloc[row_idx, :]
                cell_line = roi_row['Cell_Line']
                chrom_name = roi_row['Chr']
                seq_start = int(roi_row['Start'])
                seq_end = int(roi_row['Stop'])
                meta_row = meta_table[((meta_table['Cell_Line'] == cell_line) & (meta_table['TF'] == train_tf))]
                meta_row = meta_row.reset_index(drop=True)
                try:
                    signal = meta_row.loc[0, 'ATAC_Signal_File']
                    binding = meta_row.loc[0, 'Binding_File']
                except:
                    print("ROI-Row Val-gen")
                    print(roi_row)

                with \
                        safe_load_bigwig(filters) as filters_stream, \
                        load_bigwig(average) as average_stream, \
                        load_2bit(sequence) as sequence_stream, \
                        load_bigwig(signal) as signal_stream, \
                        load_bigwig(binding) as binding_stream:
                    input_matrix = get_input_matrix(
                        rows=INPUT_CHANNELS,
                        cols=INPUT_LENGTH,
                        batch_size=1,  # we will combine into batch later
                        reshape=False,
                        bp_order=BP_ORDER,
                        signal_stream=signal_stream,
                        average_stream=average_stream,
                        sequence_stream=sequence_stream,
                        chrom=chrom_name,
                        start=seq_start,
                        end=seq_end,
                        scale_signal=None,
                        filters_stream=filters_stream
                    )
                    inputs_batch.append(input_matrix)
                    if not quant:
                        target_vector = np.array(binding_stream.values(chrom_name, seq_start, seq_end)).T
                        target_vector = np.nan_to_num(target_vector, 0.0)
                        n_bins = int(target_vector.shape[0] / bp_resolution)
                        split_targets = np.array(np.split(target_vector, n_bins, axis=0))
                        bin_sums = np.sum(split_targets, axis=1)
                        bin_vector = np.where(bin_sums > 0.5 * bp_resolution, 1.0, 0.0)
                        targets_batch.append(bin_vector)
                    else:
                        target_vector = np.array(binding_stream.values(chrom_name, seq_start, seq_end)).T
                        target_vector = np.nan_to_num(target_vector, 0.0)
                        n_bins = int(target_vector.shape[0] / bp_resolution)
                        split_targets = np.array(np.split(target_vector, n_bins, axis=0))
                        bin_vector = np.mean(split_targets,
                                             axis=1)  # Perhaps we can change np.mean to np.median. Something to think about.
                        targets_batch.append(bin_vector)

            if quant:
                targets_batch = np.array(targets_batch)
                targets_batch = targets_batch * QUANT_TARGET_SCALE_FACTOR
            yield (np.array(inputs_batch), np.array(targets_batch))