#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# header metadata
__author__ = 'Joon Hwan Hong'
__email__ = 'joon.hong@mail.mcgill.ca'

# imports
import pandas as pd
import argparse

#TODO: fix so that it saves as region-cell instead of region_cell

# functions
def get_args():
    parser = argparse.ArgumentParser()
    # metadata files
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-m', '--metadata', type=str, required=True)
    parser.add_argument('-s', '--specimenID', type=str, required=True)
    # brain regions
    parser.add_argument('--BM10', type=str, required=True)
    parser.add_argument('--BM22', type=str, required=True)
    parser.add_argument('--BM36', type=str, required=True)
    parser.add_argument('--BM44', type=str, required=True)
    return parser.parse_args()


def concat(r1, r2, r3, r4):
    return (pd.concat([pd.read_csv(r1), pd.read_csv(r2), pd.read_csv(r3), pd.read_csv(r4)])
            .rename(columns={'Unnamed: 0': 'specimenID'}))


def pipeline(df, df2, df3, df4, list_drop, dict_repl):
    # clean up df2
    condition1 = (df2['exclude'] == 0)
    condition2 = (-df2['individualID'].isin(['Unknown', '']))
    condition3 = (df2['specimenID'].isin(df3['specimenID']))

    df2 = (df2[condition1 & condition2 & condition3]
           .drop(columns=['exclude'])
           .astype('string'))

    # clean up df1
    df = (df
          .dropna(axis=1)
          .drop(list_drop, axis=1)
          .replace(dict_repl)
          .astype('string')
          .merge(df2, on='individualID', how='inner')
          .merge(df4, on='specimenID', how='inner'))

    return df


def create_pyebm_input(df, flag_save=True):
    # separate individual region-cell values
    df['BrodmannArea'] = (df['BrodmannArea']
                          .replace({'10.0': 'frontal_pole', '22.0': 'superior_temporal_gyrus',
                                    '36.0': 'parahippocampal_gyrus', '44.0': 'inferior_frontal_gyrus'}))
    data = []
    for i in range(len(df.groupby(by='individualID'))):
        val_cell_devolution = [*df.groupby(by='individualID')][i][1][['individualID', 'BrodmannArea', 'ast', 'end', 'mic', 'neu', 'oli', 'opc']]
        temp = [*val_cell_devolution.groupby(by='BrodmannArea')]
        dict_rename2 = {'ast': f'{temp[0][0]}_ast', 'end': f'{temp[0][0]}_end',
                       'mic': f'{temp[0][0]}_mic', 'neu': f'{temp[0][0]}_neu',
                       'oli': f'{temp[0][0]}_oli', 'opc': f'{temp[0][0]}_opc'}
        df_base = temp[0][1].drop(columns=['BrodmannArea']).rename(columns=dict_rename2)
        if len(temp) > 1:
            # first case
            dict_rename3 = {'ast': f'{temp[1][0]}_ast', 'end': f'{temp[1][0]}_end',
                           'mic': f'{temp[1][0]}_mic', 'neu': f'{temp[1][0]}_neu',
                           'oli': f'{temp[1][0]}_oli', 'opc': f'{temp[1][0]}_opc'}
            df_insert = (df_base
                         .merge((temp[1][1]
                                 .drop(columns=['BrodmannArea'])
                                 .rename(columns=dict_rename3)), on='individualID'))
            # greater length
            if len(temp) > 2:
                for temps in temp[2:]:
                    dict_rename = {'ast': f'{temps[0]}_ast', 'end': f'{temps[0]}_end',
                                   'mic': f'{temps[0]}_mic', 'neu': f'{temps[0]}_neu',
                                   'oli': f'{temps[0]}_oli', 'opc': f'{temps[0]}_opc'}
                    df_insert = (df_insert
                                 .merge((temps[1]
                                         .drop(columns=['BrodmannArea'])
                                         .rename(columns=dict_rename)), on='individualID'))
            data.append(df_insert)
        else:
            data.append(df_base)
    df_values = pd.concat(data, ignore_index=True).fillna(0)

    # now create matrix for individual covariance values
    df_covars = df[['individualID', 'sex', 'race', 'ageDeath', 'pmi', 'CERAD', 'Braak', 'CDR']].drop_duplicates()

    # merge
    return df_covars.merge(df_values, on='individualID', how='inner')


# main block
def main():
    # TODO: changes to be loading a JSON file with variables to be loaded in argparse
    # load data
    args = get_args()
    list_drop = ['individualIdSource', 'species', 'ethnicity']
    dict_repl = {'90+': '90', 'male': 'Male', 'female': 'Female', 'W': '0', 'B': '1', 'H': '2', 'A': '3', 'U': '4'}
    name_output = 'DataIn_wang2018.csv'

    # preprocess BRETIGEA results
    df_bretigea = concat(
        r1=args.BM10,
        r2=args.BM22,
        r3=args.BM36,
        r4=args.BM44
    )

    # pipeline operation & save file prepared for pyebm
    df = pipeline(
        df=pd.read_csv(args.input),
        df2=pd.read_csv(args.metadata),
        df3=pd.read_csv(args.specimenID),
        df4=df_bretigea,
        list_drop=list_drop,
        dict_repl=dict_repl
    )

    # create matrix for Pyebm & final adjustments (i.e. change 'sex' --> 'Sex', and add PTID column)
    df_DataIn = (create_pyebm_input(df)
                 .drop(columns=['individualID'])
                 .rename(columns={'sex': 'Sex', 'CDR': 'Diagnosis'})
                 .replace({'Diagnosis': {'0.0': 'CN', '0.5': 'MCI',
                                         '1.0': 'AD', '2.0': 'AD', '3.0': 'AD', '4.0': 'AD', '5.0': 'AD'}}))
    df_DataIn['PTID'] = df_DataIn.index + 1
    df_DataIn['EXAMDATE'] = '2022-05-06'
    df_DataIn.to_csv(name_output, index=False)


if __name__ == "__main__":
    main()
