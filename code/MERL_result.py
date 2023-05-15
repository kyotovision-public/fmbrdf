import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob

RMSE_filename = 'MERL_fitting_result/LogRMSE.pickle'
with open(RMSE_filename, 'rb') as f:
    dict_RMSE = pickle.load(f)
sorted_list = sorted(dict_RMSE.items(), key=lambda x: x[1], reverse=True)

x = []
y = []
for item in sorted_list:
    x.append(item[0])
    y.append(item[1])

plt.figure(figsize=(18,7), dpi=300)
plt.plot(x, np.array(y)*np.log(10), marker='.', c='r', label='Ours')
plt.xticks(rotation=90, fontfamily='Times New Roman')
plt.ylabel('LogRMSE')
plt.legend(prop={'family':'Times New Roman', 'size':18})
plt.subplots_adjust(bottom=0.3)
plt.savefig('MERL_fitting_result/MERL_error_RGB.pdf')
plt.close()