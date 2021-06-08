# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 18:17:36 2020

@author: Veronica Melican
"""

import argparse
import pandas as pd
import json
from datetime import datetime, timedelta

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
        '--metadatapath',
        type=str,
        required = True,
        help = "Suffix of metadata input csv file. Beginning of this file should be the modality.\
            EXAMPLE: If input csvs are text_input.csv and call_input.csv, enter _input.csv for this parameter."
    )
    arg_parser.add_argument(
        '--completionpath',
        type=str,
        required = True,
        help = "File name of survey completion date csv."
    )
    arg_parser.add_argument(
        '--phqpath',
        type=str,
        required = True,
        help = "File name of phq score csv."
    )
    arg_parser.add_argument(
        '--output',
        type=str,
        required = True,
        help = "Name of folder to save extracted features into.\
            WARNING: If this folder already exists and has prior features from this code, they will be overwritten."
    )   
    arg_parser.add_argument(
        '--drop',
        type=int,
        required = True,
        help = "Drop time series with texts/call duration in minutes below this number."
    )
    arg_parser.add_argument(
        '--cutoff',
        type=int,
        required=True,
        help = "Cutoff to use for depression diagnosis."
    )
    return arg_parser

def main(args):
    modalities = args.modalities
    intervals = args.intervals
    directions = args.directions
    days = args.days
    metadatapath = args.metadatapath
    completionpath = args.completionpath
    phqpath = args.phqpath
    output = args.output
    drop = args.drop
    cutoff = args.cutoff
    
    for modality in modalities:
        for direction in directions:
            for day in days:
                for interval in intervals:
                    print(modality, direction, interval, day)
                    extract_time_series(modality, modality + metadatapath, completionpath, phqpath, direction, day, interval, output, drop, cutoff)

#extract time series data
def extract_time_series(modality, path, completion_path, phq_path, direction, days, interval, output, drop, cutoff):
    
    #read csv files
    content = pd.read_csv(path)
    content = content[['id','content']]
    date = pd.read_csv(completion_path)
    date = date[['id','date']]
    date.columns = ['id', 'completed']
    phq = pd.read_csv(phq_path)
    phq = phq[['id','content']]
    
    #drop duplicate metadata
    content = content.drop_duplicates(subset="content", keep="last")
        
    #calculate phq summary scores
    phq['phq_sum'] = phq['content'].apply(lambda x: calc_score(x))
    phq.drop(['content'],axis=1,inplace=True)
    
    #if there are duplicate phq scores, keep highest one
    phq = pd.DataFrame(phq.groupby('id')['phq_sum'].max()).reset_index()
    
    #extract data from text/call dataframe
    content['type'] = content['content'].apply(lambda x: get_data(x, "type"))
    if (modality=='text'):
        content['length'] = content['content'].apply(lambda x: len(get_data(x, "body")))
    elif (modality=='call'):
        content['length'] = content['content'].apply(lambda x: int(get_data(x, "duration")))
    content['date'] = content['content'].apply(lambda x: int(get_data(x, "date")))
    content = content[content['date'] != -1]
           
    #convert dates from UNIX time
    content['date'] = content['date'].apply(lambda x: datetime.fromtimestamp(x / 1000))
    date['completed'] = date['completed'].apply(lambda x: datetime.fromtimestamp(x / 1000))
    
    #merge date dataframe into content dataframe
    content = pd.merge(content, date, on='id')
    
    #round down completion date to beginning of day
    #take content within period before completion date, based on days input
    content['completed'] = content['completed'].apply(lambda x: x.replace(microsecond=0,second=0,minute=0,hour=0))
    content['start'] = content['completed'] - timedelta(days=days)
    content['is_within'] = (content['date'] < content['completed']) & (content['date'] > content['start'])
    content = content[content['is_within']]
    content.drop(['completed','start','is_within'],axis=1,inplace=True)
    
    #if direction is IN or OUT, filter based on text/call type
    if (modality == 'text'):
        if (direction == 'In'):
            content = content[content['type'] == "1"]
        elif (direction == 'Out'):
            content = content[content['type'] == "2"]
    elif (modality == 'call'):
        if (direction == 'In'):
            content = content[content['type'] != "2"]
        elif (direction == 'Out'):
            content = content[content['type'] == "2"]       
     
    #drop subjects with fewer than drop texts or fewer than drop minutes of calls
    if (modality == 'text'):
        count = content.groupby('id').count().reset_index()
        ids = count[count['content'] < drop]['id']
        content = content[~content['id'].isin(ids)]
    elif (modality == 'call'):
        count = content.groupby('id')['length'].sum().reset_index()
        count.columns = ['id', 'count']
        ids = count[count['count'] < drop * 60]['id']
        content = content[~content['id'].isin(ids)]

    #create time series vectors
    ids = []
    time_series_count = []
    time_series_average = []
    time_series_percent_zero = []
    time_series_contacts= []
    grouped = content.groupby('id')
    for name, group in grouped:
        completed = date[date['id'] == name].iloc[0]['completed']
        if modality == 'text':
            count,average,unique_contacts = get_series(modality, group, completed, days, interval)
        elif modality == 'call':
            count,average,percent_zero,unique_contacts = get_series(modality, group, completed, days, interval)
            time_series_percent_zero.append(percent_zero)
        ids.append(name)
        time_series_count.append(count)
        time_series_average.append(average)
        time_series_contacts.append(unique_contacts)
        
    time_series = pd.DataFrame()
    time_series['id'] = ids
    time_series['counts'] = time_series_count
    time_series['average_length'] = time_series_average
    if modality == 'call': time_series['percent_zero'] = time_series_percent_zero
    time_series['unique_contacts'] = time_series_contacts
    
    time_series = pd.merge(time_series, phq, on='id')
    time_series['depressed'] = time_series['phq_sum'].apply(lambda x: x >= cutoff)
        
    #create output folder if it doesn't exist already
    import os
    if not os.path.exists(output):
        os.makedirs(output)
    
    #sort csv file by id and save to csv
    time_series_sorted = time_series.sort_values(by=['id'], ascending=True)
    time_series_sorted.to_csv(output + '/' + modality + direction + str(days) + '_' + str(interval) + '.csv', index=False)

#given the metadata for phq or gad, calculate the sum
def calc_score(scores):
    sum = 0
    scores = scores[1:-1].split(",")
    for item in scores:
        sum += int(item[-2])
    return sum

#helper function that extracts data from metadata
def get_data(content, field):
    dictionary = json.loads(content);
    return dictionary[field]; 

#create time series data
#for texts: count, average length (in characters), unique contacts
#for calls: total duration, percent zero duration, average duration (in seconds), unique contacts
def get_series(modality, content, completion_date, days, interval):
    count = []
    total = []
    average = []
    percent_zero = []
    unique_contacts = []
    completion_date = completion_date.replace(microsecond = 0, second = 0, minute = 0, hour = 0)
    start = completion_date - timedelta(days=days)
    while (True):
        end = start + timedelta(hours=interval)
        temp = content[(content['date'] >= start) & (content['date'] < end)]
        if (modality == 'text'):
            #count
            count.append(temp.shape[0])
            
            #average length
            lengths = temp['content'].apply(lambda x: len(get_data(x, 'body')))
            if temp.shape[0] == 0: average.append(0)
            else: average.append(lengths.sum() / temp.shape[0])
            
            #unique contacts
            contacts = temp['content'].apply(lambda x: get_data(x, 'address'))
            unique_contacts.append(len(contacts.value_counts()))
        elif (modality == 'call'):
            #total duration
            lengths = temp['content'].apply(lambda x: int(get_data(x, 'duration')))
            lengths_non_zero = lengths[lengths != 0]
            total.append(lengths_non_zero.sum())
            
            #average duration
            if lengths_non_zero.shape[0] == 0: average.append(0)
            else: average.append(lengths_non_zero.sum() / lengths_non_zero.shape[0])
            
            #percent zero duration
            if temp.shape[0] == 0: percent_zero.append(0)
            else: percent_zero.append((temp.shape[0] - lengths_non_zero.shape[0])/temp.shape[0]) 
            
            #unique contacts
            contacts = temp['content'].apply(lambda x: get_data(x, 'number'))               
            unique_contacts.append(len(contacts.value_counts()))
        
        start = end
        end = end + timedelta(hours=interval)
        if start == completion_date: break
    if (modality == 'call'): return total, average, percent_zero, unique_contacts
    return count, average, unique_contacts

if __name__=="__main__":
    args = parser().parse_args()
    main(args)