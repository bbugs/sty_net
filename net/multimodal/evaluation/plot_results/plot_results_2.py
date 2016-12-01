import pandas
import matplotlib.pyplot as plt
import numpy as np

# fname = '../data/fashion53k/result_summary.txt'
fname = '../data/fashion53k/promising_reports/for_paper_v2/summary.txt'

df = pandas.read_csv(fname, delim_whitespace=True)

# print df[df.model =='freq']

models = ['l1', 'l5', 'a1', 'unimodal']
measure = 'precision'
# measure = 'recall'
line_types = ['g8-', 'g^--', 'b8-', 'b^--',  'kx--', 'kx--']
labels = ['l1', 'l5', 'a1', 'unimodal']
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

ymin, ymax = (0., 0.40)
ax.set_ylim((ymin, ymax))
ax.set_yticks(np.arange(ymin, ymax, 0.02))
ax.set_ylabel(measure)

ax.set_title('{} at K'.format(measure))
ax.grid(True)
ax.legend(loc=1)
plt.show()


