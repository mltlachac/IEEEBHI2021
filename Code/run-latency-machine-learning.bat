python -W ignore latency-machine-learning.py ^
--numsplits 100 ^
--maxpc 5 ^
--methods KNN LR RFC SVC ^
--samplings down up ^
--inputFile latencyfeatures10 ^
--outputFile LatencyFeature