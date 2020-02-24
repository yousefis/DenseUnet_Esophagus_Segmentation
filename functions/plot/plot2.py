import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.rand(6, 4),
                 index=name_list+name_list,
                 columns=pd.Index(['A', 'B'],
                 name='Genus')).round(2)


df.plot(kind='bar',figsize=(10,4))

ax = plt.gca()
pos = []
for bar in ax.patches:
    pos.append(bar.get_x()+bar.get_width()/2.)


ax.set_xticks(pos,minor=True)
lab = []
for i in range(len(pos)):
    l = df.columns.values[i//len(df.index.values)]
    lab.append(l)

ax.set_xticklabels(lab,minor=True)
ax.tick_params(axis='x', which='major', pad=15, size=0)
plt.setp(ax.get_xticklabels(), rotation=0)
plt.xticks(xs, name_list, rotation=-310)
plt.margins(.05)
plt.subplots_adjust(bottom=0.45)
plt.show()