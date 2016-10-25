
import pandas
import matplotlib.pyplot as plt
import numpy as np

fname = '../data/fashion53k/result_summary_v2.txt'
df = pandas.read_csv(fname, delim_whitespace=True)

models = ['a0', 'a5', 'a1', 'unimodal']

measure = 'precision'
# measure = 'recall'
line_types = ['gs-', 'm^-', 'b8-', 'ko--']
labels = ['assoc.', 'local + assoc. loss', 'local loss', 'classification']
fig, ax = plt.subplots()
i = 0
freq_vote_indicator = False  # indicates when frequency vote has already been plotted
for model in models:

    a = df[(df.model == model)]

    y = a.precision
    x = a.recall

    x = x.as_matrix()
    y = y.as_matrix()
    # label = '{}_{}'.format(model, vis)
    label = labels[i]
    line_type = line_types[i]

    ax.plot(x, y, line_type, alpha=1, label=label)

    i += 1

ax.set_xticks(np.arange(0, 1.05, 0.1))
ax.set_yticks(np.arange(0, 0.45, 0.02))

ax.grid(True)
ax.legend(loc=1)

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision vs. Recall')
ax.grid(True)
ax.legend()
plt.show()

