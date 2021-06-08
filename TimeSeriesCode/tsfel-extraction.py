import tsfel
import pandas as pd
import numpy as np
import argparse
import os

def parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--modalities',
        type=str,
        nargs = '+',
        required = True,
        help = "List of modalities to extract features for. \
            Currently supports 'text' and 'call'."
    )
    arg_parser.add_argument(
        '--directions',
        type=str,
        nargs = '+',
        required = True,
        help = "List of directions to extract features for. \
            Can be of 'All', 'In', or 'Out'."
    )
    arg_parser.add_argument(
        '--days',
        type=int,
        nargs = '+',
        required = True,
        help = "List of lengths of time series data in days."
    )
    arg_parser.add_argument(
        '--intervals',
        type=int,
        nargs = '+',
        required = True,
        help = "List of lengths of time series intervals in hours."
    )
    arg_parser.add_argument(
        '--variables',
        type=str,
        nargs = "+",
        required = True,
        help = "List of variables."
    )
    arg_parser.add_argument(
        '--inputPath',
        type=str,
        required=True,
        help = "Path to folder containing time series data."
    )
    arg_parser.add_argument(
        '--outputPath',
        type=str,
        required = True,
        help = "Path to folder to save extracted features to."
    )   
    return arg_parser

def convert_to_float_array(a):
    result = []
    for element in a:
        result.append(float(element))
    return np.asarray(result)

def main(args):
    modalities = args.modalities
    directions = args.directions
    days = args.days
    intervals = args.intervals
    variables = args.variables
    inputPath = args.inputPath
    outputPath = args.outputPath
    
    #extract features for each variable individually
    for variable in variables:
        for modality in modalities:
            for direction in directions:
                for day in days:
                    for interval in intervals:
                        #load dataset
                        df = pd.read_csv(inputPath + "/" + modality + direction + str(day) + "_" + str(interval) + ".csv")
                        depressed = df['depressed']
                        
                        ids = df['id']
                        
                        #parse
                        df[variable] = df[variable].apply(lambda x: x[1:-1])
                        df[variable] = df[variable].apply(lambda x: x.split(','))
                        df[variable] = df[variable].apply(lambda x: convert_to_float_array(x))
                        df = df[variable]
                        
                        # Retrieves a pre-defined feature configuration file to extract all available features
                        cfg = tsfel.get_features_by_domain()
                        
                        # Extract features
                        X = tsfel.time_series_features_extractor(cfg, df)
                        X['depressed'] = depressed
                        X['id'] = ids
                        
                        #create output folder if it doesn't exist already
                        if not os.path.exists(outputPath):
                            os.makedirs(outputPath)
                        
                        X.to_csv(outputPath + "/" + modality + direction + str(day) + "_" + str(interval) + "_" + variable + ".csv", index=False)
        
    #create csv files with features combined
    for modality in modalities:
        for direction in directions:
            for day in days:
                for interval in intervals:
                    #get csv for each feature
                    features = []
                    depressed = pd.DataFrame();
                    for variable in variables:
                        df = pd.read_csv(outputPath + "/" + modality + direction + str(day) + "_" + str(interval) + "_" + variable + ".csv")
                        depressed = df['depressed']
                        df.drop(['depressed'], axis=1, inplace=True)
                        features.append(df)
                    combined = pd.concat(features, axis=1)
                    combined['depressed'] = depressed
                    combined.to_csv(outputPath + "/" + modality + direction + str(day) + "_" + str(interval) + "_" + "combined" + ".csv", index=False)
                        
if __name__=="__main__":
    args = parser().parse_args()
    main(args)