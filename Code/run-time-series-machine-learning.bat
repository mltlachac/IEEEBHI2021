python -W ignore time-series-machine-learning.py ^
--modalities call text ^
--directions All In Out  ^
--days 14 ^
--intervals 4 6 12 24 ^
--variables counts average_length unique_contacts ^
--samplings up down ^
--numsplits 100 ^
--inputPath TimeSeries ^
--outputFile TimeSeriesResults


