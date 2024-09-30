import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import f

def read_data(filepath):
    data = pd.read_csv(filepath, delimiter = '\t')
    cleaned_data = data[data['EntrezGeneID'] != 0]
    cleaned_data = cleaned_data.dropna()
    zero_counts = (cleaned_data.iloc[:, 1:49] == 0).sum().sum()
    means = cleaned_data.iloc[:, 1:49].mean()
    #convert from log2 to linear scale
    cleaned_data.iloc[:, 1:49] = 2 ** cleaned_data.iloc[:, 1:49]
    return data


def analyze_data(data):
    male_nonsmokers = data.iloc[:, 106:118].values
    male_smokers = data.iloc[:, 118:130].values
    female_nonsmokers = data.iloc[:, 130:142].values
    female_smokers = data.iloc[:, 142:154].values

    # Number of groups and samples
    groups = [male_nonsmokers, male_smokers, female_nonsmokers, female_smokers]
    n_samples_per_group = [len(g) for g in groups]
    total_samples = sum(n_samples_per_group)
    n_groups = len(groups)
    group_means = [np.mean(g) for g in groups]
    overall_mean = np.mean(data)
    return group_means, overall_mean, groups, n_samples_per_group, total_samples, n_groups

def compute_f_stat(group_means, overall_mean, groups, n_samples_per_group, total_samples, n_groups):

    ssb = sum(n * (group_mean - overall_mean) ** 2 for n, group_mean in zip(n_samples_per_group, group_means))
  
    ssw = sum(sum((x - group_mean) ** 2 for x in g) for g, group_mean in zip(groups, group_means))
    
    dfb = n_groups - 1 
    dfw = total_samples - n_groups  
  
    msb = ssb / dfb
    msw = ssw / dfw
    
    f_stat = msb / msw
    return f_stat, dfb, dfw


def compute_p_value(f_stat, dfb, dfw):
  
    p_value = 1 - f.cdf(f_stat, dfb, dfw)  # Right tail of the F-distribution
    return p_value


def compute_f_Val(data):
    f_stats = []
    p_values = []

    for i in range(len(data)):
        row_data = data.iloc[i, 106:154].values  # Extract row data
        group_means, overall_mean, groups, n_samples_per_group, total_samples, n_groups = analyze_data(row_data)
        f_stat, dfb, dfw = compute_f_stat(group_means, overall_mean, groups, n_samples_per_group, total_samples, n_groups)
        p_value = compute_p_value(f_stat, dfb, dfw)
        
        f_stats.append(f_stat)
        p_values.append(p_value)

# Convert to numpy arrays for further analysis
    f_stats = np.array(f_stats)
    p_values = np.array(p_values)
    significant_rows = np.where(p_values < 0.05)[0]

    return f_stats, p_values,significant_rows

def construct_matrices():
    A_null =np.zeros((48,4)) 
    A = np.zeros((48,4))
    for i in range(48):
        A_null[i][i // 24] = 1
        A_null[i][2 + (i // 12) % 2] = 1
    for i in range(48):
        A[i][i // 12] = 1
    return A_null,A

def compute_deg_freedom(A_null, A):
    dfb = np.linalg.matrix_rank(A) 
    dfw = np.linalg.matrix_rank(A_null)
    scaling_factor = (48-dfb) / (dfb-dfw)
    return scaling_factor,dfb,dfw

def compute_frac(A,A_null):
    I = np.identity(48)
    A_null_transpose = A_null.T
    A_null_product = np.matmul(A_null_transpose, A_null)
    A_null_inv = np.linalg.pinv(A_null_product)
    A_null_inv_product = np.matmul(A_null, A_null_inv)
    A_null_inv_product_transpose = np.matmul(A_null_inv_product, A_null_transpose)
    numerator = I - A_null_inv_product_transpose

    A_transpose = A.T
    A_product = np.matmul(A_transpose, A)
    A_inv = np.linalg.pinv(A_product)
    A_inv_product = np.matmul(A, A_inv)
    A_inv_product_transpose = np.matmul(A_inv_product, A_transpose)
    denominator = I - A_inv_product_transpose

    return numerator,denominator


def compute_f_stats(A_null, A, data, scaling_factor):
    fstats=[]
    numerator,denominator = compute_frac(A,A_null)
    for _, row in data.iterrows():
        row_data = row.iloc[105:153]
        exponentiated_values = row.iloc[1:49] ** 2
        temp = np.array(exponentiated_values.to_numpy().tolist())
        temp_transpose = temp.T
        x1 = np.matmul(temp_transpose, numerator)
        x2 = np.matmul(temp_transpose, denominator)
        
        numerator_result = np.matmul(x1, temp)
        denominator_result = np.matmul(x2, temp)
        f_statistic = ((numerator_result / (denominator_result + 1e-9)) - 1) * scaling_factor
        
        fstats.append(f_statistic)
    fstats = np.array(fstats)
    fstats = fstats.tolist()
    return fstats


def compute_p(data):
    A_null, A = construct_matrices()
    scaling_factor,dfb,dfw = compute_deg_freedom(A_null, A)
    F_statistics = compute_f_stats(A_null, A, data, scaling_factor)
    p_values = 1 - stats.f.cdf(F_statistics, dfb-dfw, 48-dfb)
    return p_values


def plot_histogram(p_values):
    plt.hist(p_values, bins=50, edgecolor='k', alpha=0.7)
    plt.title('Histogram of p-values')
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.savefig('histogram.png')


def get_interesting_rows(data):
    p_values = compute_p(data)
    significant_rows = np.where(p_values < 0.05)[0]

    interesting_rows = data.iloc[significant_rows]
    interesting_rows.to_csv('interesting_genes.csv', index=False)