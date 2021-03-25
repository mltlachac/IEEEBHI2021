# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 09:58:34 2020

@author: Veronica Melican

"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from fastdtw import fastdtw

def parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--modalities', type=str, nargs = '+', required = True,
        help = "List of modalities to use, either 'text' or 'call'")
    arg_parser.add_argument('--directions', type=str, nargs = '+', required = True,
        help = "List of directions to use, either 'All', 'In', or 'Out'")
    arg_parser.add_argument('--days', type=int, nargs = '+', required = True,
        help = "List of lengths of time series data in days")
    arg_parser.add_argument('--intervals', type=int, nargs = '+', required = True,
        help = "List of lengths of time series intervals in hours")
    arg_parser.add_argument('--variables', type=str, nargs = "+", required = True,
        help = "List of variables")
    arg_parser.add_argument('--samplings', type=str, nargs = "+", required = True, choices = ['up', 'down'],
        help = "List of data balancing techniques to use, either 'up' or 'down'")
    arg_parser.add_argument('--numsplits', type=int, required = True,
        help = "Number of train test splits to run for each set of parameters")
    arg_parser.add_argument('--inputPath', type=str, required = True,
        help = "Path to the folder with input time series")
    arg_parser.add_argument('--outputFile', type=str, required = True,
        help = "Name to use for results files")
    return arg_parser
    
def main(args):
    modalities = args.modalities          
    directions = args.directions
    days = args.days
    intervals = args.intervals
    variables = args.variables
    samplings = args.samplings
    numsplits = args.numsplits
    inputPath = args.inputPath
    outputFile = args.outputFile
    
    results = []
    
    for modality in modalities:
        for direction in directions:
            for day in days:
                for variable in variables:
                    for sampling in samplings:
                        #print(modality, direction, day, variable, sampling)
                        results += run(modality, direction, day, intervals, variable, sampling, numsplits, inputPath)
                            
    #combine results into dataframes         
    combined = pd.concat(results)
    combined.to_csv(outputFile + "Results.csv");

    #create summary scores for each parameter set
    summary = pd.DataFrame();
    grouped = combined.groupby(['modality','direction','day','interval','variable','sampling'])
    
    summary['f1_mean'] = grouped['f1'].mean()
    summary['f1_stdev'] = grouped['f1'].std()
    summary['accuracy_mean'] = grouped['accuracy'].mean()
    summary['accuracy_stdev'] = grouped['accuracy'].std()
    summary['precision_mean'] = grouped['precision'].mean()
    summary['precision_stdev'] = grouped['precision'].std()
    summary['sensitivity_mean'] = grouped['sensitivity'].mean()
    summary['sensitivity_stdev'] = grouped['sensitivity'].std()
    summary['specificity_mean'] = grouped['specificity'].mean()
    summary['specificity_stdev'] = grouped['specificity'].std()
    summary['auc_mean'] = grouped['auc'].mean()
    summary['auc_stdev'] = grouped['auc'].std()
    
    summary_sorted = summary.sort_values(by='f1_mean', ascending=False)
    summary_sorted.to_csv(outputFile + "Summary.csv")


def run(modality, direction, day, intervals, variable, sampling, numsplits, inputPath):
    
    #read file to use to create indices for kfold split
    df = pd.read_csv(inputPath + '/' + modality + direction + str(day) + '_' + str(intervals[0]) + '.csv')
    df = df[['id',variable]]
    
    splits = StratifiedShuffleSplit(n_splits=numsplits, test_size=.33, random_state=100)
    splits.get_n_splits(df)
    
    results = []
    
    #iterate through intervals and k_values
    for interval in intervals:
            
        #read files
        df = pd.read_csv(inputPath + '/' + modality + direction  + str(day) + '_' + str(interval) + '.csv')
        df = df[['id',variable,'depressed']]
        
        #convert counts from string to array
        df[variable] = df[variable].apply(lambda x: x[1:-1])
        df[variable] = df[variable].apply(lambda x: x.split(','))
        df[variable] = df[variable].apply(lambda x: convert_to_float_array(x))
    
        #split into features and target
        X = df[variable]
        y = df['depressed']
        
        f1_list = []
        accuracy_list = []
        precision_list = []
        sensitivity_list= []
        specificity_list = []
        auc_list = []
        ytrue_list = []
        yhat_list = []
        splitnum_list = []
        modality_list = []
        direction_list = []
        day_list= []
        interval_list = []
        variable_list = []
        sampling_list = []

        #run for each split
        count = 0
        for train_index, test_index in splits.split(X, y):
            print(count)
            count += 1
            #print(count, interval)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
    
            #downsample/upsample
            if sampling == "up":
                X_train, y_train = upsample(X_train, y_train)
            if sampling == "down":
                X_train, y_train = downsample(X_train, y_train)
            X_train = X_train[variable]
            X_train = X_train.tolist()
            y_train = y_train.to_list()
            
            #train model
            clf = KNeighborsClassifier(metric=fastDTW, n_neighbors=3)
            clf.fit(X_train, y_train)
            
            #test model
            y_pred = clf.predict(list(X_test))
            
            #get scoring metrics
            f1, accuracy, precision, sensitivity, specificity, auc = get_scores(y_test, y_pred)
            f1_list.append(f1)
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            sensitivity_list.append(sensitivity)
            specificity_list.append(specificity)
            auc_list.append(auc)
            ytrue_list.append(y_test.values)
            yhat_list.append(y_pred)
            
            #add parameter information
            splitnum_list.append(count)
            modality_list.append(modality)
            direction_list.append(direction)
            day_list.append(day)
            interval_list.append(interval)
            variable_list.append(variable)
            sampling_list.append(sampling)
        
        #create dataframe of results
        result = pd.DataFrame()
        result['modality'] = modality_list
        result['direction'] = direction_list
        result['day'] = day_list
        result['interval'] = interval_list
        result['variable'] = variable_list
        result['sampling'] = sampling_list
        result['split_number'] = splitnum_list
        result['f1'] = f1_list
        result['accuracy'] = accuracy_list
        result['precision'] = precision_list
        result['sensitivity'] = sensitivity_list
        result['specificity'] = specificity_list
        result['auc'] = auc_list
        result['ytrue'] = ytrue_list
        result['yhat'] = yhat_list
        
        results.append(result.copy())
            
    return results

def convert_to_float_array(a):
    result = []
    for element in a:
        result.append(float(element))
    return np.asarray(result)

def downsample(X_train, y_train):
        
    X_train = pd.DataFrame(X_train)
    train = X_train.copy()
    train['y'] = y_train
    
    rus = RandomUnderSampler()
    X_train, y_train = rus.fit_sample(
        train.drop('y', axis=1), 
        train['y'])
    
    return X_train, y_train  

def upsample(X_train, y_train):
    
    X_train = pd.DataFrame(X_train)
    train = X_train.copy()
    train['y'] = y_train
    
    ros = RandomOverSampler()
    X_train, y_train = ros.fit_sample(
        train.drop('y', axis=1), 
        train['y'])
    
    return X_train, y_train

def fastDTW(a, b):
    return fastdtw(a,b)[0]  

def get_scores(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    f1 = 2*tp / (2*tp + fp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (fp + tn)
    auc = roc_auc_score(y_true, y_pred)
    
    return f1, accuracy, precision, sensitivity, specificity, auc

if __name__=="__main__":
    args = parser().parse_args()
    main(args)
