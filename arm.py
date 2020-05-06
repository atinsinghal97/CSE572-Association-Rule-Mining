import os
import sys
import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings("ignore")

def debug():
    # print (ib1_max, ib2_max, ib3_max, ib4_max, ib5_max)
    # print (cgm1_max, cgm2_max, cgm3_max, cgm4_max, cgm5_max)
    # print (cgm1_food, cgm2_food, cgm3_food, cgm4_food, cgm5_food)
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


# Setting DataFolder

if (len(sys.argv)) == 1:
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    directory = path + os.sep + 'DataFolder' + os.sep
    # directory = "/Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/DataFolder/"
    print ("Using the default path for Data Folder: ", directory)
    if (input ("Continue? (y/n)") == 'n'):
        print("Run the code as python <file-name.py> <Path-to-DataFolder>")
        print(
            "Eg: python arm.py /Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/DataFolder/")
        print ("Make sure to use the os separator at the end. It's ", os.sep, " for your OS.")
        sys.exit(0)
elif (len(sys.argv)) == 2:
    directory = sys.argv[1]
    if directory[-1]!=os.sep:
        directory = directory + os.sep
else:
    print ("Error. Run the code as python <file-name.py> <Path-to-DataFolder>")
    print ("Eg: python arm.py /Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/DataFolder/")
    print ("Make sure to use the os separator at the end. It's ", os.sep, " for your OS.")
    sys.exit(0)


dfForDataLoad = pd.DataFrame()
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


# ib1 = pd.read_csv(directory + ibFiles[0], usecols=[*range(0, 30)])
# ib2 = pd.read_csv(directory + ibFiles[1], usecols=[*range(0, 30)])
# ib3 = pd.read_csv(directory + ibFiles[2], usecols=[*range(0, 30)])
# ib4 = pd.read_csv(directory + ibFiles[3], usecols=[*range(0, 30)])
# ib5 = pd.read_csv(directory + ibFiles[4], usecols=[*range(0, 30)])
#
# cgm1 = pd.read_csv(directory + cgmFiles[0], usecols=[*range(0, 30)])
# cgm2 = pd.read_csv(directory + cgmFiles[1], usecols=[*range(0, 30)])
# cgm3 = pd.read_csv(directory + cgmFiles[2], usecols=[*range(0, 30)])
# cgm4 = pd.read_csv(directory + cgmFiles[3], usecols=[*range(0, 30)])
# cgm5 = pd.read_csv(directory + cgmFiles[4], usecols=[*range(0, 30)])


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


# Verifying data integrity

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

# print('Data verification completed successfully.')
debug()


# Binning

