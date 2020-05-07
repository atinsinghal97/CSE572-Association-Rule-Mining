Code by ATIN SINGHAL.
ASU ID# 1217358454.


How to compile code?
--------------------
This code uses Python3. The python code file imports the following libraries:
1. numpy
2. pandas
3. mlxtend
4. csv
5. os
6. sys
7. warnings


They can be installed by using the following commands:
1. pip install numpy
2. pip install pandas
3. pip install mlxtend

# csv, os, sys & warnings are included in the original Python installation.


Run the training code as "python arm.py <Path-to-DataFolder> <no-of-columns-for-output-1-or-3>"
Eg: python train.py /Users/atinsinghal97/Documents/Projects/ASU/Spring 20/CSE 572 Data Mining/Assignment 4/DataFolder/ 1

No. of columns can have a value of 1 or 3.
1 generates CSV in the format {19,6}->1.6 and (1.2,4,10)
3 generates CSV in the format 19, 6, 1.6 and 1.2, 4, 10 separated in 3 columns

Note# If your system has both Python2 & Python3, use pip3 instead of pip & python3 instead of python to compile.


Contents of Folder Submitted
-----------------------------

Files: 
1. arm.py- PYTHON Code File for generating frequent itemsets & rules
2. readme.txt


Folders:
1. DataFolder- This is the original data folder that was provided.

2. Output CSV- 1 Column Format: This folder contains 4 CSV generated in the 1-column format 	# details in line 29
	a. 1-Frequent-OneCol.csv : (1) Contains the frequent itemsets for each of the subjects (with length=3)
	b. 1-MostFrequent-OneCol.csv : (1) Contains the most frequent itemsets for each of the subjects  (with length = 3, max(support))
	c. 2-OneCol.csv : (2) Contains the rule with the largest confidence for each subject.
	d. 3-OneCol.csv : (3) Contains anomalous events by finding the least confidence rules.

3. Output CSV- 3 Column Format: This folder contains 4 CSV generated in the 3-column format 	# details in line 30
	a. 1-Frequent-MultipleCols.csv : (1) Contains the frequent itemsets for each of the subjects (with length=3)
	b. 1-MostFrequent-MultipleCols.csv : (1) Contains the most frequent itemsets for each of the subjects  (with length = 3, max(support))
	c. 2-MultipleCols.csv : (2) Contains the rule with the largest confidence for each subject.
	d. 3-MultipleCols.csv : (3) Contains anomalous events by finding the least confidence rules.