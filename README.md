# IEEEBHI2021

Materials for papers titled 'Mobile Depression Screening with Time Series of Text Logs and Call Logs' (https://ieeexplore.ieee.org/abstract/document/9508582) and 'Screening for Suicidal Ideation with Text Messages' (https://ieeexplore.ieee.org/abstract/document/9508486)

For information about depression screening from text messages, see the repositories:
https://github.com/mltlachac/IEEEjBHI2020,
https://github.com/mltlachac/IEEEBHI2019, and
https://github.com/mltlachac/EMBC2020.
More information about our depression screening research is available at https://emutivo.wpi.edu/

If you use code, data, results, or visualizations from this repository, please cite the relevant paper:

M. L. Tlachac, Veronica Melican, Miranda Reisch, Elke Rundensteiner, 'Mobile Depression Screening with Time Series of Text Logs and Call Logs', in IEEE BHI, 2021

```
@INPROCEEDINGS{TlachacTimeSeries2020,
author={Tlachac, M. L. and Melican, Veronica and Reisch, Miranda and Rundensteiner, Elke},
title={Mobile Depression Screening with Time Series of Text Logs and Call Logs},
booktitle={IEEE BHI},
year={2021}
}
```

M. L. Tlachac, Katherine Dixon-Gordon, Elke Rundensteiner,
'Screening for Suicidal Ideation with Text Messages', in IEEE BHI, 2021

```
@INPROCEEDINGS{TlachacTimeSeries2020,
author={Tlachac, M. L. and Dixon-Gordon, Katherine and Rundensteiner, Elke},
title={Screening for Suicidal Ideation with Text Messages},
booktitle={IEEE BHI},
year={2021}
}
```

## How to use the time series code

### time-series-extraction.py
This file constructs the time series from text logs and call logs. 

Eleven command-line arguments are requierd:
* modalities: either 'call' or 'text'
* directions: either 'All', 'In' or 'Out'
* days: number of days within survey completion date to use data from
* intervals: aggregation intervals for time series
* drop: drop participants with fewer than this number of texts or minutes of calls
* cutoff: PHQ-9 cutoff for depression diagnosis
* metadatapath: suffix of .csv file containing text or call logs
    * For example, if logs are stored in 'text_input.csv' and 'call_input.csv', put '_input.csv' for this argument
    * This .csv file has two columns: one with participant id and another with text log or call log metadata in json format
* completionpath: .csv file containing survey completion dates
    * This .csv file has two columns: one with participant id and another with survey completion date in UNIX time
* phqpath: .csv file containing PHQ-9 question responses
    * This .csv file contains two columns: one with participant id and another with PHQ-9 question responses in json format
    * Example: Example: {"Q0":"2","Q1":"1","Q2":"2","Q3":"1","Q4":"2","Q5":"3","Q6":"2","Q7":"1","Q8":"2"}
* output: name of folder to output time series to

Lists can be input for the modalities, directions, days, and intervals arguments and the code will create time series with all combinations of the inputs.

View run-time-series-extraction.bat for an example of how these arguments are specified.

Note that the required .csv files must be in the same directory as this code.

### tsfel-extraction.py
This file extracts features from the time series constructed with time-series-extraction.py using the TSFEL library.

Seven command-line arguments are required:
* modalities: either 'call' or 'text'
* directions: either 'All', 'In' or 'Out'
* days: number of days within survey completion date to use data from
* intervals: aggregation intervals for time series
* variables: either 'counts', 'average_length' or 'unique_contacts'
    * An additional variable, called combined, will be created that combines features for all variables input into one .csv file
* inputPath: path to folder containing time series
* outputPath: path to folder to output features to

Lists can be input for the modalities, directions, days, intervals, and variables arguments and the code will extract features with all combinations of the inputs.

View run-tsfel-extraction.bat for an example of how these arguments are specified.

### time-series-machine-learning.py
This file runs machine learning experiments on the time series constructed with time-series-extraction.py using k-Nearest Neighbors and dynamic time warping distance.

Nine command-line arguments are required:
* modalities: either 'call' or 'text'
* directions: either 'All', 'In' or 'Out'
* days: number of days within survey completion date to use data from
* intervals: aggregation intervals for time series
* variables: either 'counts', 'average_length' or 'unique_contacts'
* samplings: either 'up' or 'down'
* numsplits: number of train/test splits to run for each combination of arguments
* inputPath: path to folder containg time series
* outputFile: name of file to output result to
    * There will be two output files, one ending in 'Results' that has the results of each individual train/test split, and another ending in 'Summary' that has the mean and standard deviation of the results for each combination of arguments
    
Lists can be input for the modalities, directions, days, intervals, variables, and samplings arguments and the code will run experiments with all combinations of the inputs.

View run-time-series-machine-learning.bat for an example of how these arguments are specified.

### feature-machine-learning.py
This file runs machine learning experiments on the time series features.

Eleven command-line arguments are required:
* modalities: either 'call' or 'text'
* directions: either 'All', 'In' or 'Out'
* days: number of days within survey completion date to use data from
* intervals: aggregation intervals for time series
* variables: either 'counts', 'average_length' or 'unique_contacts'
* samplings: either 'up' or 'down'
* methods: either 'LR', 'RFC', 'SVC' or 'KNN'
* numsplits: number of train/test splits to run for each combination of arguments
* maxpc: maximum number of principal components to use, the code will iterate from one up to this number
* inputPath: path to folder containing features
* outputFile: name of file to output result to
    * There will be two output files, one ending in 'Results' that has the results of each individual train/test split, and another ending in 'Summary' that has the mean and standard deviation of the results for each combination of arguments

Lists can be input for the modalities, directions, days, intervals, variables, samplings, and methods arguments and the code will run experiments with all combinations of the inputs. 

View run-feature-machine-learning.bat for an example of how these arguments are specified.

### latency-machine-learning.py
This file runs machine learning experiments with the latency features.

Six command-line arguments are required:
* samplings: either 'up' or 'down'
* methods: either 'LR', 'RFC', 'SVC' or 'KNN'
* numsplits: number of train/test splits to run for each combination of arguments
* maxpc: maximum number of principal components to use, the code will iterate from one up to this number
* inputFile: path to file containing features
* outputFile: name of file to output result to
    * There will be two output files, one ending in 'Results' that has the results of each individual train/test split, and another ending in 'Summary' that has the mean and standard deviation of the results for each combination of arguments
    
Lists can be input for the methods and samplings arguments and the code will run experiments with all combinations of the inputs.

View run-latency-machine-learning.bat for an example of how these arguments are specified.

### ResultVisualization.ipynb
This notebook creates visualizations of the results. 

The first cell creates visualizations for the experiments with time series features.

The second cell creates visualizations for the experiments with time series.

The path to the file containing results, variable to visualize, and plot title can be specified at the top of main() in each cell.
