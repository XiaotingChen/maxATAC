from deeplift.dinuc_shuffle import dinuc_shuffle  # function to do a dinucleotide shuffle
import numpy as np
from multiprocessing import Pool
from functools import partial

from maxatac.utilities.constants import INPUT_CHANNELS, INPUT_LENGTH, BP_ORDER
from maxatac.utilities.genome_tools import safe_load_bigwig, load_bigwig, load_2bit
from maxatac.utilities.training_tools import get_input_matrix


def dinuc_shuffle_DNA_only_several_times(list_containing_input_modes_for_an_example,
                                         seed=1234):
    assert len(list_containing_input_modes_for_an_example) == 1

    onehot_seq = list_containing_input_modes_for_an_example[0][:, :4]  # DNA only: length x 4
    ATAC_signals = list_containing_input_modes_for_an_example[0][:, 4:]  # ATAC-seq signal only: length x 2

    rng = np.random.RandomState(seed)
    to_return = np.array([np.concatenate((dinuc_shuffle(onehot_seq, rng=rng),
                                          ATAC_signals), axis=1) for _ in range(10)])
    return [to_return]


# shap explainer's combine_mult_and_diffref function
def combine_DNA_only_mult_and_diffref(mult, orig_inp, bg_data):

    assert len(orig_inp) == 1
    projected_hypothetical_contribs = np.zeros_like(bg_data[0]).astype("float")

    assert len(orig_inp[0].shape) == 2

    for i in range(4):
        hypothetical_input = np.zeros_like(orig_inp[0]).astype("float")  # length x 6  all set to 0
        hypothetical_input[:, i] = 1.0  # change only DNA position
        hypothetical_input[:, -2:] = orig_inp[0][:, -2:]  # copy over ATAC signal
        hypothetical_difference_from_reference = (hypothetical_input[None, :, :] - bg_data[0])
        hypothetical_contribs = hypothetical_difference_from_reference * mult[0]
        projected_hypothetical_contribs[:, :, i] = np.sum(hypothetical_contribs, axis=-1)

    return [np.mean(projected_hypothetical_contribs, axis=0)]


def output_meme_pwm(pwm, pattern_name):
    with open(pattern_name + '.meme', 'w') as f:
        f.writelines('MEME version 4\n')
        f.writelines('\n')
        f.writelines('ALPHABET= ACGT\n')
        f.writelines('\n')
        f.writelines('strands: + -\n')
        f.writelines('\n')
        f.writelines('Background letter frequencies (from uniform background):\n')
        f.writelines('A 0.25000 C 0.25000 G 0.25000 T 0.25000 \n')
        f.writelines('\n')

        l = np.shape(pwm)[0]

        f.writelines('MOTIF {} {}\n'.format(pattern_name, pattern_name))
        f.writelines('\n')
        f.writelines('letter-probability matrix: alength= 4 w= {} nsites= 1 E= 0\n'.format(l))

        for i in range(0, l):
            _sum = np.sum([pwm[i, 0], pwm[i, 1], pwm[i, 2], pwm[i, 3]])
            f.writelines('  {}	  {}	  {}	  {}	\n'.format(float(pwm[i, 0]) / _sum, float(pwm[i, 1]) / _sum,
                                                                    float(pwm[i, 2]) / _sum, float(pwm[i, 3]) / _sum))
        f.writelines('\n')


def generating_interpret_data(sequence,
                              meta_table,
                              roi_pool,
                              train_tf,
                              bp_resolution=1,
                              workers=8
                              ):
    _mp = Pool(workers)
    _data = np.array(_mp.map(partial(process_map,
                                     sequence=sequence,
                                     meta_table=meta_table,
                                     roi_pool=roi_pool,
                                     train_tf=train_tf,
                                     bp_resolution=bp_resolution,
                                     ), roi_pool.index[:])
                     )
    _mp.close()
    _mp.join()
    return _data[:, 0], _data[:, 1]


def process_map(row_idx,
                sequence,
                meta_table,
                roi_pool,
                train_tf,
                bp_resolution):
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
            load_2bit(sequence) as sequence_stream, \
            load_bigwig(signal) as signal_stream, \
            load_bigwig(binding) as binding_stream:

        input_matrix = get_input_matrix(
                rows=INPUT_CHANNELS,
                cols=INPUT_LENGTH,
                bp_order=BP_ORDER,
                signal_stream=signal_stream,
                sequence_stream=sequence_stream,
                chromosome=chrom_name,
                start=start,
                end=end
            )
        target_vector = np.array(binding_stream.values(chrom_name, start, end)).T
        target_vector = np.nan_to_num(target_vector, 0.0)

        n_bins = int(target_vector.shape[0] / bp_resolution)
        split_targets = np.array(np.split(target_vector, n_bins, axis=0))

        bin_sums = np.sum(split_targets, axis=1)
        bin_vector = np.where(bin_sums > 0.5 * bp_resolution, 1.0, 0.0)

    return [input_matrix, bin_vector]


def get_roi_pool(filepath, chroms, shuffle=False):
    """
    Import the ROI file containing the regions of interest. This file is similar to a bed file, but with a header

    The roi DF is read in from a TSV file that is formatted similarly as a BED file with a header. The following columns
    are required:

    Chr | Start | Stop | ROI_Type | Cell_Line

    The chroms list is used to filter the ROI df to make sure that only training chromosomes are included.

    :param chroms: A list of chromosomes to filter the ROI pool by. This is a double check that it is prefiltered
    :param filepath: The path to the roi file to be used
    :param shuffle: Whether to shuffle the dataframe upon import

    :return: A pool of regions to use for training or validation
    """
    roi_df = pd.read_csv(filepath, sep="\t", header=0, index_col=None)

    roi_df = roi_df[roi_df['Chr'].isin(chroms)]

    if shuffle:
        roi_df = roi_df.sample(frac=1)

    return roi_df
