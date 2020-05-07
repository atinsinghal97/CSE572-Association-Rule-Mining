import os
import sys
import numpy as np
import pandas as pd
import csv

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

import warnings

warnings.filterwarnings("ignore")


no_of_columns = 0


# Debug Functions

def debug(print_=False):
    debug_cgm_max(print_)
    debug_cgm_food(print_)
    debug_ib(print_)
    # debug_dataset(print_)
    # debug_frequentitemset(print_)
    # debug_rules(print_)

def debug_ib(print_=False):
    if print_ is not False:
        print (ib1_max, ib2_max, ib3_max, ib4_max, ib5_max)
    ib1_max.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/ib1_max.csv",
        index=False, header=False)
    ib2_max.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/ib2_max.csv",
        index=False, header=False)
    ib3_max.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/ib3_max.csv",
        index=False, header=False)
    ib4_max.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/ib4_max.csv",
        index=False, header=False)
    ib5_max.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/ib5_max.csv",
        index=False, header=False)

def debug_cgm_max(print_=False):
    if print_ is not False:
        print (cgm1_max, cgm2_max, cgm3_max, cgm4_max, cgm5_max)
    cgm1_max.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/cgm1_max.csv",
        index=False, header=False)
    cgm2_max.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/cgm2_max.csv",
        index=False, header=False)
    cgm3_max.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/cgm3_max.csv",
        index=False, header=False)
    cgm4_max.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/cgm4_max.csv",
        index=False, header=False)
    cgm5_max.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/cgm5_max.csv",
        index=False, header=False)

def debug_cgm_food(print_=False):
    if print_ is not False:
        print (cgm1_food, cgm2_food, cgm3_food, cgm4_food, cgm5_food)
    cgm1_food.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/cgm1_food.csv",
        index=False, header=False)
    cgm2_food.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/cgm2_food.csv",
        index=False, header=False)
    cgm3_food.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/cgm3_food.csv",
        index=False, header=False)
    cgm4_food.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/cgm4_food.csv",
        index=False, header=False)
    cgm5_food.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/cgm5_food.csv",
        index=False, header=False)

def debug_dataset(print_=False):
    if print_ is not False:
        print (dataset1, dataset2, dataset3, dataset4, dataset5)
    dataset1.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/dataset1.csv",
        index=False, header=False)
    dataset2.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/dataset2.csv",
        index=False, header=False)
    dataset3.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/dataset3.csv",
        index=False, header=False)
    dataset4.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/dataset4.csv",
        index=False, header=False)
    dataset5.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/dataset5.csv",
        index=False, header=False)

def debug_frequentitemset(print_=False):
    if print_ is not False:
        print (frequent_itemsets1, frequent_itemsets2, frequent_itemsets3, frequent_itemsets4, frequent_itemsets5)
    head_bool=True
    frequent_itemsets1.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/frequent_itemsets1.csv",
        index=False, header=head_bool)
    frequent_itemsets2.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/frequent_itemsets2.csv",
        index=False, header=head_bool)
    frequent_itemsets3.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/frequent_itemsets3.csv",
        index=False, header=head_bool)
    frequent_itemsets4.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/frequent_itemsets4.csv",
        index=False, header=head_bool)
    frequent_itemsets5.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/frequent_itemsets5.csv",
        index=False, header=head_bool)

def debug_rules(print_=False):
    if print_ is not False:
        print (rules1, rules2, rules3, rules4, rules5)
    head_bool=True
    rules1.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/rules1.csv",
        index=False, header=head_bool)
    rules2.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/rules2.csv",
        index=False, header=head_bool)
    rules3.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/rules3.csv",
        index=False, header=head_bool)
    rules4.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/rules4.csv",
        index=False, header=head_bool)
    rules5.to_csv(
        r"/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/Processed Data/rules5.csv",
        index=False, header=head_bool)


# Setting DataFolder

if (len(sys.argv)) == 1:
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    directory = path + os.sep + 'DataFolder' + os.sep
    # directory = "/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/DataFolder/"
    print ("Using the default path for Data Folder: ", directory)
    if (input ("Continue? (y/n)") == 'n'):
        print("Run the code as python <file-name.py> <Path-to-DataFolder> <no-of-columns-for-output-1-or-3>")
        print(
            "Eg: python arm.py /Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/DataFolder/ 1")
        print ("Make sure to use the os separator at the end. It's ", os.sep, " for your OS.")
        sys.exit(0)
elif (len(sys.argv)) == 3:
    no_of_columns = int(sys.argv[2])
    if (no_of_columns == 1 or no_of_columns == 3) is False:
        print("No. of columns can have a value of 1 or 3.")
        print("1 generates CSV in the format {19,6}->1.6 and (1.2,4,10)")
        print("3 generates CSV in the format 19, 6, 1.6 and 1.2, 4, 10 separated in 3 columns")
        print(no_of_columns, " is not a valid value. Try Again with 1 or 3.")
        sys.exit(0)
    directory = sys.argv[1]
    if directory[-1]!=os.sep:
        directory = directory + os.sep
else:
    print ("Error. Run the code as python <file-name.py> <Path-to-DataFolder> <no-of-columns-for-output-1-or-3>")
    print ("Eg: python arm.py /Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/DataFolder/ 1")
    print ("Make sure to use the os separator at the end. It's ", os.sep, " for your OS.")
    sys.exit(0)


# List of Files

ibFiles = ['InsulinBolusLunchPat1.csv', 'InsulinBolusLunchPat2.csv', 'InsulinBolusLunchPat3.csv', 'InsulinBolusLunchPat4.csv', 'InsulinBolusLunchPat5.csv']
cgmFiles = ['CGMSeriesLunchPat1.csv', 'CGMSeriesLunchPat2.csv', 'CGMSeriesLunchPat3.csv', 'CGMSeriesLunchPat4.csv', 'CGMSeriesLunchPat5.csv']


# Read CSV Files

ib1 = pd.read_csv(directory + ibFiles[0])
ib2 = pd.read_csv(directory + ibFiles[1])
ib3 = pd.read_csv(directory + ibFiles[2])
ib4 = pd.read_csv(directory + ibFiles[3])
ib5 = pd.read_csv(directory + ibFiles[4])

cgm1 = pd.read_csv(directory + cgmFiles[0])
cgm2 = pd.read_csv(directory + cgmFiles[1])
cgm3 = pd.read_csv(directory + cgmFiles[2])
cgm4 = pd.read_csv(directory + cgmFiles[3])
cgm5 = pd.read_csv(directory + cgmFiles[4])


# Check CGM data for NaN values & dropping 'em

check_nan_cgm1 = cgm1['cgmSeries_25'].isnull()
check_nan_cgm2 = cgm2['cgmSeries_25'].isnull()
check_nan_cgm3 = cgm3['cgmSeries_25'].isnull()
check_nan_cgm4 = cgm4['cgmSeries_25'].isnull()
check_nan_cgm5 = cgm5['cgmSeries_25'].isnull()

for i in range(0, len(cgm1)):
    if check_nan_cgm1[i] is np.bool_(True):
        # print(i, "True")
        cgm1 = cgm1.drop(index=i)
        ib1 = ib1.drop(index=i)
cgm1.reset_index(drop=True, inplace=True)
ib1.reset_index(drop=True, inplace=True)
# print (cgm1)

for i in range(0, len(cgm2)):
    if check_nan_cgm2[i] is np.bool_(True):
        # print(i, "True")
        cgm2 = cgm2.drop(index=i)
        ib2 = ib2.drop(index=i)
cgm2.reset_index(drop=True, inplace=True)
ib2.reset_index(drop=True, inplace=True)
# print (cgm2)

for i in range(0, len(cgm3)):
    if check_nan_cgm3[i] is np.bool_(True):
        # print(i, "True")
        cgm3 = cgm3.drop(index=i)
        ib3 = ib3.drop(index=i)
cgm3.reset_index(drop=True, inplace=True)
ib3.reset_index(drop=True, inplace=True)
# print (cgm3)

for i in range(0, len(cgm4)):
    if check_nan_cgm4[i] is np.bool_(True):
        # print(i, "True")
        cgm4 = cgm4.drop(index=i)
        ib4 = ib4.drop(index=i)
cgm4.reset_index(drop=True, inplace=True)
ib4.reset_index(drop=True, inplace=True)
# print (cgm4)

for i in range(0, len(cgm5)):
    if check_nan_cgm5[i] is np.bool_(True):
        # print(i, "True")
        cgm5 = cgm5.drop(index=i)
        ib5 = ib5.drop(index=i)
cgm5.reset_index(drop=True, inplace=True)
ib5.reset_index(drop=True, inplace=True)
# print (cgm5)


# Getting ib data ready

ib1_max = ib1.max(axis=1)
ib2_max = ib2.max(axis=1)
ib3_max = ib3.max(axis=1)
ib4_max = ib4.max(axis=1)
ib5_max = ib5.max(axis=1)

# print (ib1_max, ib2_max, ib3_max, ib4_max, ib5_max)
# print (ib2_max)


# Checking ib data for NaN values & dropping 'em

check_nan_ib1_max = ib1_max.isnull()
check_nan_ib2_max = ib2_max.isnull()
check_nan_ib3_max = ib3_max.isnull()
check_nan_ib4_max = ib4_max.isnull()
check_nan_ib5_max = ib5_max.isnull()
# print (check_nan_ib2_max)

for i in range(0, len(ib1_max)):
    if check_nan_ib1_max[i] is np.bool_(True):
        # print(i, "True")
        cgm1 = cgm1.drop(index=i)
        ib1_max = ib1_max.drop(index=i)
cgm1.reset_index(drop=True, inplace=True)
ib1_max.reset_index(drop=True, inplace=True)
# print (cgm1, ib1_max)

for i in range(0, len(ib2_max)):
    if check_nan_ib2_max[i] is np.bool_(True):
        # print(i, "True")
        cgm2 = cgm2.drop(index=i)
        ib2_max = ib2_max.drop(index=i)
cgm2.reset_index(drop=True, inplace=True)
ib2_max.reset_index(drop=True, inplace=True)
# print (cgm2, ib2_max)

for i in range(0, len(ib3_max)):
    if check_nan_ib3_max[i] is np.bool_(True):
        # print(i, "True")
        cgm3 = cgm3.drop(index=i)
        ib3_max = ib3_max.drop(index=i)
cgm3.reset_index(drop=True, inplace=True)
ib3_max.reset_index(drop=True, inplace=True)
# print (cgm3, ib3_max)

for i in range(0, len(ib4_max)):
    if check_nan_ib4_max[i] is np.bool_(True):
        # print(i, "True")
        cgm4 = cgm4.drop(index=i)
        ib4_max = ib4_max.drop(index=i)
cgm4.reset_index(drop=True, inplace=True)
ib4_max.reset_index(drop=True, inplace=True)
# print (cgm4, ib4_max)

for i in range(0, len(ib5_max)):
    if check_nan_ib5_max[i] is np.bool_(True):
        # print(i, "True")
        cgm5 = cgm5.drop(index=i)
        ib5_max = ib5_max.drop(index=i)
cgm5.reset_index(drop=True, inplace=True)
ib5_max.reset_index(drop=True, inplace=True)
# print (cgm5, ib5_max)

# print('Pre-processing completed successfully.')


# Getting cgm_max data ready

cgm1_max = cgm1.max(axis=1)
cgm2_max = cgm2.max(axis=1)
cgm3_max = cgm3.max(axis=1)
cgm4_max = cgm4.max(axis=1)
cgm5_max = cgm5.max(axis=1)
# print(cgm1_max, cgm2_max, cgm3_max, cgm4_max, cgm5_max)


# Getting cgm_food data ready

cgm1_food = cgm1['cgmSeries_25']
cgm2_food = cgm2['cgmSeries_25']
cgm3_food = cgm3['cgmSeries_25']
cgm4_food = cgm4['cgmSeries_25']
cgm5_food = cgm5['cgmSeries_25']
# print (cgm1_food, cgm2_food, cgm3_food, cgm4_food, cgm5_food)


# Data sanity check

if (cgm1_food.shape == cgm1_max.shape == ib1_max.shape) is False:
    print ("Issues with Patient 1 Data")
    sys.exit(0)
if (cgm2_food.shape == cgm2_max.shape == ib2_max.shape) is False:
    print ("Issues with Patient 2 Data")
    sys.exit(0)
if (cgm3_food.shape == cgm3_max.shape == ib3_max.shape) is False:
    print ("Issues with Patient 3 Data")
    sys.exit(0)
if (cgm4_food.shape == cgm4_max.shape == ib4_max.shape) is False:
    print ("Issues with Patient 4 Data")
    sys.exit(0)
if (cgm5_food.shape == cgm5_max.shape == ib5_max.shape) is False:
    print ("Issues with Patient 5 Data")
    sys.exit(0)

# print('Data sanity check completed successfully.')
# debug(print_=False)


# Binning

def getBin(val):
    # if val>40 and val<=50:
    #     return 1
    # elif val>50 and val<=60:
    #     return 2
    if float(val)%10 == float(0):
        return int(val/10)-4
    else:
        return int(val/10)-3

cgm1_max_bin = cgm1_max
cgm2_max_bin = cgm2_max
cgm3_max_bin = cgm3_max
cgm4_max_bin = cgm4_max
cgm5_max_bin = cgm5_max

cgm1_food_bin = cgm1_food
cgm2_food_bin = cgm2_food
cgm3_food_bin = cgm3_food
cgm4_food_bin = cgm4_food
cgm5_food_bin = cgm5_food

cgm1_max_bin = cgm1_max_bin.apply(lambda x: getBin(x))
cgm2_max_bin = cgm2_max_bin.apply(lambda x: getBin(x))
cgm3_max_bin = cgm3_max_bin.apply(lambda x: getBin(x))
cgm4_max_bin = cgm4_max_bin.apply(lambda x: getBin(x))
cgm5_max_bin = cgm5_max_bin.apply(lambda x: getBin(x))
# print(cgm1_max_bin, cgm1_max)

cgm1_food_bin = cgm1_food_bin.apply(lambda x: getBin(x))
cgm2_food_bin = cgm2_food_bin.apply(lambda x: getBin(x))
cgm3_food_bin = cgm3_food_bin.apply(lambda x: getBin(x))
cgm4_food_bin = cgm4_food_bin.apply(lambda x: getBin(x))
cgm5_food_bin = cgm5_food_bin.apply(lambda x: getBin(x))


# Converting df to list to get filtered outputs

cgm_max_bin_list = list()
cgm_max_bin_list.append(cgm1_max_bin.to_list())
cgm_max_bin_list.append(cgm2_max_bin.to_list())
cgm_max_bin_list.append(cgm3_max_bin.to_list())
cgm_max_bin_list.append(cgm4_max_bin.to_list())
cgm_max_bin_list.append(cgm5_max_bin.to_list())
# for x in cgm_max_bin_list:
    # print(x)


cgm_food_bin_list = list()
cgm_food_bin_list.append(cgm1_food_bin.tolist())
cgm_food_bin_list.append(cgm2_food_bin.tolist())
cgm_food_bin_list.append(cgm3_food_bin.tolist())
cgm_food_bin_list.append(cgm4_food_bin.tolist())
cgm_food_bin_list.append(cgm5_food_bin.tolist())
# for x in cgm_food_bin_list:
    # print(x)

ib_max_list = list()
ib_max_list.append(ib1_max.tolist())
ib_max_list.append(ib2_max.tolist())
ib_max_list.append(ib3_max.tolist())
ib_max_list.append(ib4_max.tolist())
ib_max_list.append(ib5_max.tolist())
# for x in ib_max_list:
    # print(x)

# Creating Dataset

dataset1 = pd.concat([cgm1_max_bin, cgm1_food_bin, ib1_max], axis=1)
dataset1.columns = ['cgm_max', 'cgm_food', 'ib']
dataset2 = pd.concat([cgm2_max_bin, cgm2_food_bin, ib2_max], axis=1)
dataset2.columns = ['cgm_max', 'cgm_food', 'ib']
dataset3 = pd.concat([cgm3_max_bin, cgm3_food_bin, ib3_max], axis=1)
dataset3.columns = ['cgm_max', 'cgm_food', 'ib']
dataset4 = pd.concat([cgm4_max_bin, cgm4_food_bin, ib4_max], axis=1)
dataset4.columns = ['cgm_max', 'cgm_food', 'ib']
dataset5 = pd.concat([cgm5_max_bin, cgm5_food_bin, ib5_max], axis=1)
dataset5.columns = ['cgm_max', 'cgm_food', 'ib']
# print (dataset1, dataset2, dataset3, dataset4, dataset5)

# print('Dataset created successfully.')
# debug_dataset(print_=False)


# Finding Frequent Itemsets & Rules

def fir(dataset):
    dataset = dataset.values.tolist()
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=10**-10, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=10**-10)
    rules = rules.drop(columns=['antecedent support', 'consequent support', 'support', 'lift', 'leverage', 'conviction'])
    return frequent_itemsets, rules

frequent_itemsets1, rules1 = fir(dataset1)
frequent_itemsets2, rules2 = fir(dataset2)
frequent_itemsets3, rules3 = fir(dataset3)
frequent_itemsets4, rules4 = fir(dataset4)
frequent_itemsets5, rules5 = fir(dataset5)

# print('Itemsets & Rules found successfully.')
# debug_frequentitemset(print_=False)
# print(frequent_itemsets1.shape, frequent_itemsets2.shape, frequent_itemsets3.shape, frequent_itemsets4.shape, frequent_itemsets5.shape)
# debug_rules(print_=False)


# Finding Most Frequent Itemsets

def mfi(frequent_itemsets):
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    frequent_itemsets = frequent_itemsets[frequent_itemsets['length'] == 3]
    len3_itemsets = frequent_itemsets
    len3_itemsets.reset_index(drop=True, inplace=True)
    max_support = frequent_itemsets['support'].max()
    frequent_itemsets = frequent_itemsets[frequent_itemsets['support'] == max_support]
    frequent_itemsets.reset_index(drop=True, inplace=True)
    # print('Max Support: ', max_support)
    # print(frequent_itemsets)
    return frequent_itemsets, len3_itemsets

mfi1, len3_itemsets1 = mfi(frequent_itemsets1)
mfi2, len3_itemsets2 = mfi(frequent_itemsets2)
mfi3, len3_itemsets3 = mfi(frequent_itemsets3)
mfi4, len3_itemsets4 = mfi(frequent_itemsets4)
mfi5, len3_itemsets5 = mfi(frequent_itemsets5)
# print(mfi1.shape, mfi2.shape, mfi3.shape, mfi4.shape, mfi5.shape)
# print(len3_itemsets1)


# Filtering Rules

def filter_rules(rules, i):
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequent_len"] = rules["consequents"].apply(lambda x: len(x))

    # Selecting rules of the form A, B -> C
    rules = rules[(rules['antecedent_len'] == 2) & (rules['consequent_len'] == 1)]
    rules.reset_index(drop=True, inplace=True)

    # Selecting rules where C belongs to ib
    rules['consequents_check'] = rules['consequents'].apply(
        lambda x: 'True' if list(x)[0] in ib_max_list[i - 1] else 'False')
    rules = rules[rules['consequents_check'] == 'True']
    rules.reset_index(drop=True, inplace=True)

    # Selecting rules where A & B belong to cgm_max or cgm_food
    rules['antecedents_check'] = rules['antecedents'].apply(lambda x: 'True' if (
                (list(x)[0] in cgm_food_bin_list[i - 1] and list(x)[1] in cgm_max_bin_list[i - 1]) or (
                    list(x)[1] in cgm_food_bin_list[i - 1] and list(x)[0] in cgm_max_bin_list[i - 1])) else 'False')
    rules = rules[rules['antecedents_check'] == 'True']
    rules.reset_index(drop=True, inplace=True)

    # print(rules)

    # Selecting rules where confidence is minimum
    min_confidence = rules['confidence'].min()
    min_rules = rules[rules['confidence'] == min_confidence]
    min_rules.reset_index(drop=True, inplace=True)

    # Selecting rules where confidence is maximum
    max_confidence = rules['confidence'].max()
    max_rules = rules[rules['confidence'] == max_confidence]
    max_rules.reset_index(drop=True, inplace=True)

    # print('Min Confidence: ', min_confidence)
    # print(min_rules)
    # print('Max Confidence: ', max_confidence)
    # print(max_rules)
    return min_rules, max_rules

min_rules1, max_rules1 = filter_rules(rules1, 1)
min_rules2, max_rules2 = filter_rules(rules2, 2)
min_rules3, max_rules3 = filter_rules(rules3, 3)
min_rules4, max_rules4 = filter_rules(rules4, 4)
min_rules5, max_rules5 = filter_rules(rules5, 5)


# Getting data ready for output CSVs

def prepareDF_fi(df, param_=1): #mfi, len3_itemsets
    intermediateList1 = list()
    intermediateList2 = list()
    intermediateList3 = list()
    df['itemsets'].apply(lambda x: intermediateList1.append(list(x)[0]))
    df['itemsets'].apply(lambda x: intermediateList2.append(list(x)[1]))
    df['itemsets'].apply(lambda x: intermediateList3.append(list(x)[2]))
    # for x in zip(intermediateList1, intermediateList2, intermediateList3): print (x)
    if param_==1:
        return zip(intermediateList1, intermediateList2, intermediateList3)
    elif param_==3:
        return intermediateList1, intermediateList2, intermediateList3


def prepareDF_rules(df, param_=1): #min_riles, max_rules
    antecedentsList_1 = list()
    antecedentsList_2 = list()
    consequentsList = list()

    df['antecedents'].apply(lambda x: antecedentsList_1.append(list(x)[0]))
    df['antecedents'].apply(lambda x: antecedentsList_2.append(list(x)[1]))
    df['consequents'].apply(lambda x: consequentsList.append(list(x)[0]))
    # for x in zip(antecedentsList_1, antecedentsList_2, consequentsList): print (x)
    if param_==1:
        return zip(antecedentsList_1, antecedentsList_2, consequentsList)
    elif param_==3:
        return antecedentsList_1, antecedentsList_2, consequentsList


# Generating CSV

def make_csv_3col(zipObj1, zipObj2, zipObj3, zipObj4, zipObj5, file):
    with open(file, "w") as f:
        writer = csv.writer(f)
        for row in zipObj1:
            writer.writerow(row)
        for row in zipObj2:
            writer.writerow(row)
        for row in zipObj3:
            writer.writerow(row)
        for row in zipObj4:
            writer.writerow(row)
        for row in zipObj5:
            writer.writerow(row)


def make_csv_1col_fi(file1, file2):
    combined = list()
    a = list()
    b = list()
    c = list()
    a, b, c = prepareDF_fi(len3_itemsets1, param_=3)
    for i in range (len(a)):
        combined.append('('+str(a[i])+','+str(b[i])+','+str(c[i])+')')
    a = list()
    b = list()
    c = list()
    a, b, c = prepareDF_fi(len3_itemsets2, param_=3)
    for i in range(len(a)):
        combined.append('(' + str(a[i]) + ',' + str(b[i]) + ',' + str(c[i]) + ')')
    a = list()
    b = list()
    c = list()
    a, b, c = prepareDF_fi(len3_itemsets3, param_=3)
    for i in range(len(a)):
        combined.append('(' + str(a[i]) + ',' + str(b[i]) + ',' + str(c[i]) + ')')
    a = list()
    b = list()
    c = list()
    a, b, c = prepareDF_fi(len3_itemsets4, param_=3)
    for i in range(len(a)):
        combined.append('(' + str(a[i]) + ',' + str(b[i]) + ',' + str(c[i]) + ')')
    a = list()
    b = list()
    c = list()
    a, b, c = prepareDF_fi(len3_itemsets5, param_=3)
    for i in range(len(a)):
        combined.append('(' + str(a[i]) + ',' + str(b[i]) + ',' + str(c[i]) + ')')
    df1 = pd.DataFrame(combined)
    df1.to_csv(file1, index=False, header=False)

    combined = list()
    a = list()
    b = list()
    c = list()
    a, b, c = prepareDF_fi(mfi1, param_=3)
    for i in range(len(a)):
        combined.append('(' + str(a[i]) + ',' + str(b[i]) + ',' + str(c[i]) + ')')
    a = list()
    b = list()
    c = list()
    a, b, c = prepareDF_fi(mfi2, param_=3)
    for i in range(len(a)):
        combined.append('(' + str(a[i]) + ',' + str(b[i]) + ',' + str(c[i]) + ')')
    a = list()
    b = list()
    c = list()
    a, b, c = prepareDF_fi(mfi3, param_=3)
    for i in range(len(a)):
        combined.append('(' + str(a[i]) + ',' + str(b[i]) + ',' + str(c[i]) + ')')
    a = list()
    b = list()
    c = list()
    a, b, c = prepareDF_fi(mfi4, param_=3)
    for i in range(len(a)):
        combined.append('(' + str(a[i]) + ',' + str(b[i]) + ',' + str(c[i]) + ')')
    a = list()
    b = list()
    c = list()
    a, b, c = prepareDF_fi(mfi5, param_=3)
    for i in range(len(a)):
        combined.append('(' + str(a[i]) + ',' + str(b[i]) + ',' + str(c[i]) + ')')
    df2 = pd.DataFrame(combined)
    df2.to_csv(file2, index=False, header=False)


def make_csv_1col_rules(file1, file2):
    combined = list()
    a = list()
    b = list()
    c = list()
    a, b, c = prepareDF_rules(min_rules1, param_=3)
    for i in range (len(a)):
        combined.append('{'+str(a[i])+','+str(b[i])+'}->'+str(c[i]))
    a = list()
    b = list()
    c = list()
    a, b, c = prepareDF_rules(min_rules2, param_=3)
    for i in range(len(a)):
        combined.append('{' + str(a[i]) + ',' + str(b[i]) + '}->' + str(c[i]))
    a = list()
    b = list()
    c = list()
    a, b, c = prepareDF_rules(min_rules3, param_=3)
    for i in range(len(a)):
        combined.append('{' + str(a[i]) + ',' + str(b[i]) + '}->' + str(c[i]))
    a = list()
    b = list()
    c = list()
    a, b, c = prepareDF_rules(min_rules4, param_=3)
    for i in range(len(a)):
        combined.append('{' + str(a[i]) + ',' + str(b[i]) + '}->' + str(c[i]))
    a = list()
    b = list()
    c = list()
    a, b, c = prepareDF_rules(min_rules5, param_=3)
    for i in range(len(a)):
        combined.append('{' + str(a[i]) + ',' + str(b[i]) + '}->' + str(c[i]))
    df1 = pd.DataFrame(combined)
    df1.to_csv(file1, index=False, header=False)

    combined = list()
    a = list()
    b = list()
    c = list()
    a, b, c = prepareDF_rules(max_rules1, param_=3)
    for i in range(len(a)):
        combined.append('{' + str(a[i]) + ',' + str(b[i]) + '}->' + str(c[i]))
    a = list()
    b = list()
    c = list()
    a, b, c = prepareDF_rules(max_rules2, param_=3)
    for i in range(len(a)):
        combined.append('{' + str(a[i]) + ',' + str(b[i]) + '}->' + str(c[i]))
    a = list()
    b = list()
    c = list()
    a, b, c = prepareDF_rules(max_rules3, param_=3)
    for i in range(len(a)):
        combined.append('{' + str(a[i]) + ',' + str(b[i]) + '}->' + str(c[i]))
    a = list()
    b = list()
    c = list()
    a, b, c = prepareDF_rules(max_rules4, param_=3)
    for i in range(len(a)):
        combined.append('{' + str(a[i]) + ',' + str(b[i]) + '}->' + str(c[i]))
    a = list()
    b = list()
    c = list()
    a, b, c = prepareDF_rules(max_rules5, param_=3)
    for i in range(len(a)):
        combined.append('{' + str(a[i]) + ',' + str(b[i]) + '}->' + str(c[i]))
    df2 = pd.DataFrame(combined)
    df2.to_csv(file2, index=False, header=False)


# Save 1-column CSVs
if no_of_columns==1 or no_of_columns==0:
    make_csv_1col_fi('1-Frequent-OneCol.csv', '1-MostFrequent-OneCol.csv')
    make_csv_1col_rules('3-OneCol.csv', '2-OneCol.csv')

# Save 3-column CSVs
if no_of_columns==3 or no_of_columns==0:
    make_csv_3col(prepareDF_rules(min_rules1), prepareDF_rules(min_rules2), prepareDF_rules(min_rules3), prepareDF_rules(min_rules4), prepareDF_rules(min_rules5), '3-MultipleCols.csv')
    make_csv_3col(prepareDF_rules(max_rules1), prepareDF_rules(max_rules2), prepareDF_rules(max_rules3), prepareDF_rules(max_rules4), prepareDF_rules(max_rules5), '2-MultipleCols.csv')
    make_csv_3col(prepareDF_fi(mfi1), prepareDF_fi(mfi2), prepareDF_fi(mfi3), prepareDF_fi(mfi4), prepareDF_fi(mfi5), '1-MostFrequent-MultipleCols.csv')
    make_csv_3col(prepareDF_fi(len3_itemsets1), prepareDF_fi(len3_itemsets2), prepareDF_fi(len3_itemsets3), prepareDF_fi(len3_itemsets4), prepareDF_fi(len3_itemsets5), '1-Frequent-MultipleCols.csv')
