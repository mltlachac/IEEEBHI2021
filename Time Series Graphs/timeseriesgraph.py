import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
import csv



#define files by type and aggregation interval
file_name_all = "/Users/miranda/PycharmProjects/TimeSeriesGraphs/ExtractedFeatures/textAll14_24.csv"
file_name_in = "/Users/miranda/PycharmProjects/TimeSeriesGraphs/ExtractedFeatures/textIn14_24.csv"
file_name_out = "/Users/miranda/PycharmProjects/TimeSeriesGraphs/ExtractedFeatures/textOut14_24.csv"

#read files
df_all = pd.read_csv(file_name_all)
df_in = pd.read_csv(file_name_in)
df_out = pd.read_csv(file_name_out)

#make lists
#all
id_list_all = []
counts_list_all = []
depressed_list_all = []
average_list_all = []
contacts_list_all = []
#in
id_list_in = []
counts_list_in = []
depressed_list_in = []
average_list_in = []
contacts_list_in = []
#out
id_list_out = []
counts_list_out = []
depressed_list_out = []
average_list_out = []
contacts_list_out = []

indices_all = []
indices_in = []
indices_out = []

#append to lists
#ID
for ia in df_all["id"]:
    id_list_all.append(ia)
for ii in df_in["id"]:
    id_list_in.append(ii)
for io in df_out["id"]:
    id_list_out.append(io)

#Counts
for ca in df_all["counts"]:
    counts_list_all.append(ca)
for ci in df_in["counts"]:
    counts_list_in.append(ci)
for co in df_out["counts"]:
    counts_list_out.append(co)
#Average Length
for la in df_all["average_length"]:
    average_list_all.append(la)
for li in df_in["average_length"]:
    average_list_in.append(li)
for lo in df_out["average_length"]:
    average_list_out.append(lo)
#Unique Contacts
for ua in df_all["unique_contacts"]:
    contacts_list_all.append(ua)
for ui in df_in["unique_contacts"]:
    contacts_list_in.append(ui)
for uo in df_out["unique_contacts"]:
    contacts_list_out.append(uo)
#Depressed
for da in df_all["phq_sum"]:
    depressed_list_all.append(da)
for di in df_in["phq_sum"]:
    depressed_list_in.append(di)
for do in df_out["phq_sum"]:
    depressed_list_out.append(do)

number = 0
index_List = []
list_String_all = []
list_String_in = []
list_String_out = []

for x in id_list_all:

    String_all = []
    index_all = id_list_all.index(x)
    indices_all.append(index_all)
    String_all = counts_list_all[index_all]
    string_B = str(String_all)[1: -1]
    list_String_all = string_B.split(',')
    list_String_Final_all = list(map(int, list_String_all))
    a_LSF_all = np.asarray(list_String_Final_all)

    if x in id_list_in:
        String_in = []
        index_in = id_list_in.index(x)
        indices_in.append(index_in)
        String_in = counts_list_in[index_in]
        string_B = str(String_in)[1: -1]
        list_String_in = string_B.split(',')
        list_String_Final_in = list(map(int, list_String_in))
        a_LSF_in = np.asarray(list_String_Final_in)

    else:
        list_String_Final_in = [0] * len(list_String_Final_all)
        a_LSF_in = np.asarray(list_String_Final_in)


    if x in id_list_out:
        String_out = []
        index_out = id_list_out.index(x)
        indices_out.append(index_out)
        String_out = counts_list_out[index_out]
        string_B = str(String_out)[1: -1]
        list_String_out = string_B.split(',')
        list_String_Final_out = list(map(int, list_String_out))
        a_LSF_out = np.asarray(list_String_Final_out)

    else:
        list_String_Final_out = [0] * len(list_String_Final_all)
        a_LSF_out = np.asarray(list_String_Final_out)


    index_List = []
    number = 0
    for a in range(len(list_String_all)):

        number += 1
        index_List.append(number)

    val = len(index_List)/14
    val2 = int(val)
    seg = str(24/val)


    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 15
    plt.rcParams["figure.figsize"] = fig_size
    plt.plot(index_List, a_LSF_all, color="black", linewidth = 3, alpha = 1, marker = "s", label = "All", markersize = 6)
    plt.plot(index_List, a_LSF_in, color="blue", linewidth = 2.5, alpha = 0.75, marker = "D", label = "Incoming", markersize = 6)
    plt.plot(index_List, a_LSF_out, linewidth = 2, color = "darkorange", alpha = 0.75, marker = "o", label = "Outgoing", markersize = 6)
    plt.title("Id" + " " + x + " - " + seg + " Hour Segment over 14 Days" + " - " + "PHQ-9: " + str(depressed_list_all[index_all]))
    plt.xlabel("Day Number")
    plt.xticks(np.arange(min(index_List), max(index_List)+1, val2), ("Day 1", "Day 2", "Day 3", "Day 4", "Day 5",
                                                                       "Day 6", "Day 7", "Day 8", "Day 9", "Day 10",
                                                                       "Day 11", "Day 12", "Day 13", "Day 14"))

    plt.ylabel("Number of Texts")

    plt.legend()

    #Change for graph type and aggregation interval
    plt.savefig(str(depressed_list_all[index_all]) + "_" + x + "counts_" + "texts_24" + ".png", dpi = 300)


    #plt.show()
    plt.close()