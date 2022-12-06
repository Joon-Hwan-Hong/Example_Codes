#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# header metadata
__author__ = 'Joon Hwan Hong'
__email__ = 'joon.hong@mail.mcgill.ca'
'''
script for running sustain package on z-score matrix of wang2018 data
'''

# imports
import os
import pylab
import argparse
import pySuStaIn
import numpy as np
import pandas as pd
import dill as pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


# functions
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True)
    parser.add_argument('-o', '--out_dir', type=str, required=True)
    parser.add_argument('-n', '--out_name', type=str, required=True)
    parser.add_argument('-r', '--rerun', type=bool, required=False)
    args = parser.parse_args()
    return args


def get_data(dir_input, drop_col='EXAMDATE', idx_biomarker=8):
    # load data
    df = (pd
          .read_csv(dir_input)
          .drop(columns=drop_col))
    biomarkers = df.columns[idx_biomarker:]
    return df[biomarkers].values, biomarkers, df


def plot_outputs(sustain_object, samples_sequence, samples_f, biomarker_labels, N_S_max, output_folder, dataset_name,
                 N_iterations_MCMC):
    # plot sustain output
    m = 300  # number of observations ( e.g. subjects )
    sustain_object.plot_positional_var(samples_sequence=samples_sequence,
                                       samples_f=samples_f,
                                       m=m,
                                       Z_vals=sustain_object.Z_vals,
                                       biomarker_labels=biomarker_labels,
                                       subtype_order=(0, 1, 2, 3))
    _ = plt.figure(0)
    _ = plt.show()

    # plot MCMC trace & model likelihood histogram
    for s in range(N_S_max):
        pickle_filename_s = output_folder + '/pickle_files/' + dataset_name + '_subtype' + str(s) + '.pickle'
        pickle_file = open(pickle_filename_s, 'rb')
        loaded_variables = pickle.load(pickle_file)
        samples_likelihood = loaded_variables["samples_likelihood"]
        pickle_file.close()

        _ = plt.figure(1)
        _ = plt.plot(range(N_iterations_MCMC), samples_likelihood, label="subtype" + str(s))
        _ = plt.figure(2)
        _ = plt.hist(samples_likelihood, label="subtype" + str(s))

    _ = plt.figure(1)
    _ = plt.legend(loc='upper right')
    _ = plt.xlabel('MCMC samples')
    _ = plt.ylabel('Log likelihood')
    _ = plt.title('Figure 2: MCMC trace')

    _ = plt.figure(2)
    _ = plt.legend(loc='upper right')
    _ = plt.xlabel('Log likelihood')
    _ = plt.ylabel('Number of samples')
    _ = plt.title('Figure 3: Histograms of model likelihood')
    _ = plt.show()


def get_CVIC(sustain_object, test_idxs, N_S_max, n_splits):
    CVIC, loglike_matrix = sustain_object.cross_validate_sustain_model(test_idxs.astype(int))

    # go through each subtypes model and plot the log-likelihood on the test set and the CVIC
    _ = plt.figure()
    _ = plt.plot(np.arange(N_S_max, dtype=int), CVIC)
    _ = plt.xticks(np.arange(N_S_max, dtype=int))
    _ = plt.ylabel('CVIC')
    _ = plt.xlabel('Subtypes model')
    _ = plt.title('CVIC')
    _ = plt.show()

    _ = plt.figure()
    df_loglike = pd.DataFrame(data=loglike_matrix, columns=["s_" + str(i) for i in range(sustain_object.N_S_max)])
    df_loglike.boxplot(grid=False)
    for i in range(sustain_object.N_S_max):
        y = df_loglike[["s_" + str(i)]]
        x = np.random.normal(1 + i, 0.04, size=len(y))  # Add some random "jitter" to the x-axis
        pylab.plot(x, y, 'r.', alpha=0.2)
    _ = plt.ylabel('Log likelihood')
    _ = plt.xlabel('Subtypes model')
    _ = plt.title('Figure 8: Test set log-likelihood across folds')

    # this part estimates cross-validated positional variance diagrams
    for i in range(4):
        sustain_object.combine_cross_validated_sequences(i + 1, n_splits)

    return CVIC, loglike_matrix


# main block
def main():
    # TODO: for testing purpose - remove later
    np.random.seed(42)

    # ********** Prepare inputs for z-score SuStaIn **********
    """
    data: data to run SuStaIn on, of size M subjects by N biomarkers
    Z_vals: set of z-scores you want to include for each biomarker. Z_vals has size N biomarkers by Z z-scores
    Z_max: maximum z-score reached at the end of the progression, with size N biomarkers by 1
    biomarker_labels: names of the biomarkers for plotting purposes
    N_startpoints: number of startpoints to use when fitting the subtypes hierarchichally
    N_S_max: maximum number of subtypes to fit
    N_iterations_MCMC: number of iterations for the MCMC sampling of the uncertainty in the progression pattern
    output_folder: output folder for the results
    dataset_name: name the results files outputted by SuStaIn
    use_parallel_startpoints: Boolean for whether or not to parallelize the startpoints
    """
    # TODO: make all of this based on input params on argparse

    # get input args
    args = get_args()
    # for sustain object
    data, biomarker_labels, zdata = get_data(args.data)
    Z_vals = np.array([[1, 2, 3]] * len(biomarker_labels))
    Z_max = np.array([5] * len(biomarker_labels))
    N_startpoints = 25
    N_S_max = 4
    N_iterations_MCMC = int(1e4)
    output_folder = args.out_dir
    dataset_name = args.out_name
    use_parallel_startpoints = True
    # for cross validation
    n_splits = 10

    # create SuStaIn class
    sustain_object = pySuStaIn.ZscoreSustainMissingData(data=data,
                                                        Z_vals=Z_vals,
                                                        Z_max=Z_max,
                                                        biomarker_labels=biomarker_labels,
                                                        N_startpoints=N_startpoints,
                                                        N_S_max=N_S_max,
                                                        N_iterations_MCMC=N_iterations_MCMC,
                                                        output_folder=output_folder,
                                                        dataset_name=dataset_name,
                                                        use_parallel_startpoints=use_parallel_startpoints)

    # Create results folder if it doesn't exist
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # ********** run z-score SuStaIn **********

    (samples_sequence,
     samples_f,
     ml_subtype,
     prob_ml_subtype,
     ml_stage,
     prob_ml_stage,
     prob_subtype_stage) = sustain_object.run_sustain_algorithm()

    # ********** plotting and stuff **********
    plot_outputs(sustain_object, samples_sequence, samples_f, biomarker_labels, N_S_max, output_folder, dataset_name, N_iterations_MCMC)

    # ********** Choosing the optimal # of subtypes **********
    # The CVIC is an information criterion that balances model complexity with model accuracy,
    # a lower CVIC indicate a better balance between. Generally speaking, the model with the lowest CVIC is the best.

    # Stratified cross-validation
    labels = zdata.Diagnosis.values
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    test_idxs = np.array([test for _, test in cv.split(zdata, labels)], dtype='object')

    # perform cross-validation and output the cross-validation information criterion and
    # log-likelihood on the test set for each subtypes model and fold combination
    CVIC, loglike_matrix = get_CVIC(sustain_object=sustain_object,
                                    test_idxs=test_idxs,
                                    N_S_max=N_S_max,
                                    n_splits=n_splits)


if __name__ == "__main__":
    main()
