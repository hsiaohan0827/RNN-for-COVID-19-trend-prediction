import csv
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

rows = []
with open('covid_19.csv', newline='') as csvfile:
    rows = list(csv.reader(csvfile))
    rows = rows[3:]

# calculate daily increasing
daily_diff = []
country_list = []
for country in rows:
    diff = [ float(country[i+1]) - float(country[i]) for i in range(3, len(country)-1) ]
    diff = diff[len(diff)//2:]
    daily_diff.append(diff)
    country_list.append(country[0])

print('# of countries: {}'.format(len(country_list)))
print('# of total days: {}'.format(len(rows[3])-3))

# calculate correlation coefficient
daily_diff = np.array(daily_diff)
corr_array = np.full((len(country_list), len(country_list)), -1.)
for i in range(len(daily_diff)-1):
    for j in range(i+1, len(daily_diff)):
        corr, _ = pearsonr(daily_diff[i], daily_diff[j])
        corr_array[j][i] = corr

''' all countries '''
# figure setting
fig = plt.figure(figsize=(40,40))
ax = fig.add_subplot(111)
ax.set_yticks(range(len(country_list)))
ax.set_yticklabels(country_list)
ax.set_xticks(range(len(country_list)))
ax.set_xticklabels(country_list)
plt.xticks(rotation=45)

# set colorbar
im = ax.imshow(corr_array, cmap=plt.cm.Reds)
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=50)

# save figure
fig.savefig('corr_array.png')

''' choose training pair (corr > threshold=0.5) '''
threshold=0
training_pair = []
choose_flag = [0] * len(country_list)
for i in range(len(corr_array)-1):
    for j in range(i+1, len(corr_array)):
        if corr_array[j][i] >= threshold:
            if choose_flag[i] is 0:
                training_pair.append(daily_diff[i])
                choose_flag[i] = 1
            if choose_flag[j] is 0:
                training_pair.append(daily_diff[j])
                choose_flag[j] = 1
            
training_pair = np.array(training_pair)
choose_flag = np.array(choose_flag)
print('above threshold: {}'.format(len(training_pair)))

# normalize the sequence
for x in range(len(training_pair)):
    training_pair[x] = (training_pair[x]) / (np.std(training_pair[x]))

# save training / testing / all pair for training / testing / predicting
# save flag of countries above threshold
np.save('all_country', training_pair)
np.save('country_above-{}'.format(threshold), choose_flag)
np.random.shuffle(training_pair)
np.save('testing_pair-{}'.format(threshold), training_pair[:len(training_pair)//4])
np.save('training_pair-{}'.format(threshold), training_pair[len(training_pair)//4:])
