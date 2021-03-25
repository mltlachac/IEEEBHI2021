python -W ignore feature-machine-learning.py ^
--modalities call text ^
--directions All In Out ^
--days 14 ^
--intervals 4 6 12 24 ^
--variables average_length counts unique_contacts combined ^
--methods LR SVC KNN RFC ^
--samplings up down ^
--numsplits 100 ^
--maxpc 15 ^
--inputPath TSFELFeatures ^
--outputFile TSFELResults

