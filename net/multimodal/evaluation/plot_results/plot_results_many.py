
import pandas
import matplotlib.pyplot as plt
import numpy as np

fname = '../data/fashion53k/result_summary_v2.txt'
df = pandas.read_csv(fname, delim_whitespace=True)

models = ['a0', 'a5', 'a1', 'unimodal']

# measure = 'precision'
measure = 'recall'
line_types = ['gs-', 'm^-', 'b8-', 'ko--']
labels = ['assoc.', 'local + assoc. loss', 'local loss', 'classification']
fig, ax = plt.subplots()
i = 0
freq_vote_indicator = False  # indicates when frequency vote has already been plotted
for model in models:

    a = df[(df.model == model)]

    if measure == 'precision':
        y = a.precision
    elif measure == 'recall':
        y = a.recall
    elif measure == 'F':
        y = a.f
    else:
        raise ValueError("measure must be Precision, Recall or F1")

    x = a.K
    x = x.as_matrix()
    y = y.as_matrix()
    # label = '{}_{}'.format(model, vis)
    label = labels[i]
    line_type = line_types[i]

    ax.plot(x, y, line_type, alpha=1, label=label)

    i += 1

xmin, xmax = (0, 21)
ax.set_xlim((xmin, xmax))
ax.set_xticks(range(xmin, xmax, 2))
ax.set_xlabel('K')

ymin, ymax = (0., 0.45)
ax.set_ylim((ymin, ymax))
ax.set_yticks(np.arange(ymin, ymax, 0.02))
ax.set_ylabel(measure)

ax.set_title('{} at K'.format(measure))
ax.grid(True)
ax.legend(loc=4)
plt.show()

