'''
---------------------------------------------------------------------------
MACS 40200: Final Project
Contributors: Bobae Kang
              Dongping Zhang
Python Version: 3.5
Seed: None

This is a Python script for MACS 40200 Structural Estimation final project.
The project estimates healthcare expenditure of individuals using a Markov
regime-swtiching model.
---------------------------------------------------------------------------
'''

'''
---------------------------------------------------------------------------
Preparation:
    1. import required packages for analysis
    2. import raw data and data cleaning
---------------------------------------------------------------------------
'''
# import packages
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn
import os

# set the directory for saving images
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = 'images'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

# import the data
data = pd.read_pickle('claims_adj.pkl')

# Attention:
# there are negative expenditures in the dataset: remove them
# 1. find all negative expenditure by booleanizing the dataframe
non_zeros = (data.iloc[:,3:] < 0).sum(axis = 1) == 0
# 2. remove those rows (subjects) if the sum of those booleans != 0
data_nonneg = data[non_zeros]

# extract monthly expenditures
exp = data_nonneg.iloc[:,3:]
# numpy array of the monthly expenditures to facilitate operations
exp_arr = np.array(exp)
# training set (first 12 months)
exp_train = exp.iloc[:,:12]
# test set (second 12 months)
exp_test = exp.iloc[:,12:]

# index variable for positive monthly expenditure
pos_ix = np.zeros_like(exp_arr)
for i in list(range(0,exp_arr.shape[0])):
    for j in list(range(0,exp_arr.shape[1])):
        if exp_arr[i,j] > 0:
            pos_ix[i,j] = 1
        else:
            pos_ix[i,j] = 0

# index variable for zero monthly expenditure
zero_ix = np.zeros_like(exp_arr)
for i in list(range(0,exp_arr.shape[0])):
    for j in list(range(0,exp_arr.shape[1])):
        if exp_arr[i,j] == 0:
            zero_ix[i,j] = 1
        else:
            zero_ix[i,j] = 0

# mean and sta of the mean expenditure by the number of positive-expediture months
pos_mean = np.zeros(exp.shape[1])
pos_std = np.zeros(exp.shape[1])
pos_spell_ix = []
for i in range(exp.shape[1]):
    spell_ix = (pos_ix.sum(axis=1) == i+1)
    pos_spell_ix.append(spell_ix)
    pos_mean[i] = ((exp[spell_ix]).sum(axis=1) / (i+1)).mean()
    pos_std[i] = ((exp[spell_ix]).sum(axis=1) / (i+1)).std()

'''
---------------------------------------------------------------------------
Descriptive stats: (Any suggestions?)
---------------------------------------------------------------------------
'''
# gender
data_nonneg.male[data_nonneg.male!=True].count()
data_nonneg.male[data_nonneg.male==True].count()

# age, all
data_nonneg.age.mean()
data_nonneg.age.median()
data_nonneg.age.max()
data_nonneg.age.min()
data_nonneg.age.std()

# age, female
data_nonneg[data_nonneg.male!=True].age.mean()
data_nonneg[data_nonneg.male!=True].age.median()
data_nonneg[data_nonneg.male!=True].age.max()
data_nonneg[data_nonneg.male!=True].age.min()
data_nonneg[data_nonneg.male!=True].age.std()

# age, male
data_nonneg[data_nonneg.male==True].age.mean()
data_nonneg[data_nonneg.male==True].age.median()
data_nonneg[data_nonneg.male==True].age.max()
data_nonneg[data_nonneg.male==True].age.min()
data_nonneg[data_nonneg.male==True].age.std()

# expenditure
np.array(data_nonneg.iloc[:,3:]).shape
np.array(data_nonneg.iloc[:,3:]).mean()
np.median(np.array(data_nonneg.iloc[:,3:]))
np.array(data_nonneg.iloc[:,3:]).max()
np.array(data_nonneg.iloc[:,3:]).min()
np.array(data_nonneg.iloc[:,3:]).std()

# expenditure, female
np.array(data_nonneg[data_nonneg.male!=True].iloc[:,3:]).shape
np.array(data_nonneg[data_nonneg.male!=True].iloc[:,3:]).mean()
np.median(np.array(data_nonneg[data_nonneg.male!=True].iloc[:,3:]))
np.array(data_nonneg[data_nonneg.male!=True].iloc[:,3:]).max()
np.array(data_nonneg[data_nonneg.male!=True].iloc[:,3:]).min()
np.array(data_nonneg[data_nonneg.male!=True].iloc[:,3:]).std()

# expenditure, male
np.array(data_nonneg[data_nonneg.male==True].iloc[:,3:]).shape
np.array(data_nonneg[data_nonneg.male==True].iloc[:,3:]).mean()
np.median(np.array(data_nonneg[data_nonneg.male==True].iloc[:,3:]))
np.array(data_nonneg[data_nonneg.male==True].iloc[:,3:]).max()
np.array(data_nonneg[data_nonneg.male==True].iloc[:,3:]).min()
np.array(data_nonneg[data_nonneg.male==True].iloc[:,3:]).std()

# expenditure, zero
zero_ix.sum()

# expenditure, positive
exp_val_vec = np.array(exp).flatten()
exp_val_vec[exp_val_vec > 0].mean()
np.median(exp_val_vec[exp_val_vec > 0])
exp_val_vec[exp_val_vec > 0].max()
exp_val_vec[exp_val_vec > 0].min()
exp_val_vec[exp_val_vec > 0].std()

'''
<summary stats>

name | n | mean | median | max | min | std
---
age | 99965 | 33.20 | 36 | 95 | 0 | 18.45
age, female | 51203 | 33.30 | 36 | 94 | 0 | 18.18
age, male | 48762 | 33.09 | 36 | 95 | 0 | 18.73
monthly expenditure | 2399160 | 305.31 | 0.0 | 422855.84 | 0.0 | 2235.93
monthly expenditure, female | 1228872 | 328.18 | 0.0 | 355495.02 | 0.0 | 2008.85
monthly expenditure, male | 1170288 | 281.29 | 0.0 | 422855.84 | 0.0 | 2451.61
monthly expenditure, zero | 1375975 | . | . | . | . | .
monthly expenditure, positive | 1023185 | 715.88 | 173.30 | 422855.84 | 0.01 | 3380.62
#zero expenditure spell |
#positive expenditure spell |
'''

'''
---------------------------------------------------------------------------
Data visualization:
    1. Mean and std of positive expenditures
    2. Mean and std of mean expenditure by the number of positive-expediture month
    3. Fraction of indivdiuals with zero expenditure spells
    4. Fraction of indivdiuals with positive expenditure spells
    5. Distribution of male and female individuals by age
---------------------------------------------------------------------------
'''
months = np.array((range(exp.shape[1])))
months_train = np.array(range(exp_train.shape[1]))

# Plot 1
# plot of mean and std of expenditures
# plt.plot(months, exp.mean(), color='maroon',
#     label='Mean of monthly exp.')
# plt.plot(months, exp.std(), linestyle = '--', color = 'maroon',
#     label='Std. dev. of monthly exp.')
plt.plot(months, exp[exp!=0].mean(), color ='navy',
    label='Mean of positive monthly exp.')
plt.plot(months, exp[exp!=0].std(), linestyle = '--', color ='maroon',
    label='Std. dev. of positive monthly exp.')
plt.xlim(-.5, 23.5)
plt.ylim(0, 5000)
plt.xticks(months, months+1)
plt.title('Sample includes 99,965 individuals', fontsize=15)
plt.xlabel('Month', fontsize=13)
plt.ylabel('Monthly expenditure ($)', fontsize=13)
legend = plt.legend(loc='upper left', prop={'size':12}, handlelength=1.5, frameon=True)
frame = legend.get_frame()
frame.set_facecolor('white')

# save the plot
output_path_1 = os.path.join(output_dir, 'fig_1')
plt.savefig(output_path_1, bbox_inches = 'tight')

plt.show()
plt.close()

'''---------------------------------------------------------------------'''
# Plot 2
# plot of mean and standard deviation of mean positive expenditure by month
plt.plot(months, pos_mean, color='navy',
    label='Mean of mean pos. exp.')
plt.plot(months, pos_std, linestyle = '--', color = 'maroon',
    label='Std. dev. of mean pos. exp.')
plt.xlim(-.5, 23.5)
plt.xticks(months, months+1)
plt.title('Sample includes 99,965 individuals', fontsize=15)
plt.xlabel('Number of positive-expenditure months', fontsize=13)
plt.ylabel('Monthly expenditure ($)', fontsize=13)
legend = plt.legend(loc='upper left', prop={'size':12}, handlelength=1.5, frameon=True)
frame = legend.get_frame()
frame.set_facecolor('white')

# save the plot
output_path_2 = os.path.join(output_dir, 'fig_2')
plt.savefig(output_path_2, bbox_inches = 'tight')

plt.show()
plt.close()

'''---------------------------------------------------------------------'''
# Plot 3
# plot of fraction of indivdiuals with zero expenditure spells
plt.hist(zero_ix.sum(axis=1), bins=24, normed=True, align='left', color='goldenrod')
plt.xlim(-1, 24)
plt.title('Sample includes 99,965 individuals', fontsize=15)
plt.xlabel('Number of months with zero expenditures', fontsize=13)
plt.ylabel('Franction of individuals', fontsize=13)

# save the plot
output_path_3 = os.path.join(output_dir, 'fig_3')
plt.savefig(output_path_3, bbox_inches = 'tight')

plt.show()
plt.close()

'''---------------------------------------------------------------------'''
# Plot 4
# plot of fraction of indivdiuals with positive expenditure spells
plt.hist(pos_ix.sum(axis=1), bins=24, normed=True, align='left', color='goldenrod')
plt.xlim(-1, 24)
plt.title('Sample includes 99,965 individuals', fontsize=15)
plt.xlabel('Number of months with positive expenditures', fontsize=13)
plt.ylabel('Franction of individuals', fontsize=13)

# save the plot
output_path_4 = os.path.join(output_dir, 'fig_4')
plt.savefig(output_path_4, bbox_inches = 'tight')

plt.show()
plt.close()

'''---------------------------------------------------------------------'''
# Plot 5
# plot of the distribution of male and female individuals by age
fig = plt.figure()
# female plot
ax1 = fig.add_subplot(121)
ax1.hist(data[data.gender != 'M'].age, bins=95, label='Female', normed=True,
        color='r', alpha=0.5, orientation='horizontal')
ax1.invert_xaxis()
ax1.set_ylabel('Age', fontsize=13)
ax1.set_yticks([])
legend1 = ax1.legend(loc='upper left', fontsize=12, frameon=True)
frame1 = legend1.get_frame()
frame1.set_facecolor('white')
# male plot
ax2 = fig.add_subplot(122)
ax2.hist(data[data.gender == 'M'].age, bins=95, label='Male', normed=True,
        color='b', alpha=0.5, orientation='horizontal')
legend2 = ax2.legend(loc='upper right', fontsize=12, frameon=True)
frame2 = legend2.get_frame()
frame2.set_facecolor('white')
fig.suptitle('Sample includes 99,965 individuals', fontsize=15)
fig.text(0.5, 0.02, 'Fraction of individuals', ha='center', fontsize=13)

# save the plot
output_path_5 = os.path.join(output_dir, 'fig_5')
plt.savefig(output_path_5, bbox_inches = 'tight')

plt.show()
plt.close()
