python -W ignore tsfel-extraction.py ^
--modalities text call ^
--directions All In Out ^
--days 14 ^
--intervals 4 6 12 24 ^
--variables average_length counts unique_contacts ^
--inputPath TimeSeries ^
--outputPath TSFELFeatures

