import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_style("ticks")
sns.set_context("poster",font_scale=1.28,rc={"lines.linewidth": 3})

### plot robustness result
noise = np.array([0,0.2,0.4,0.6,0.8,1.0])
MLP_degree = np.array([0.3440, 0.1365, 0.0663, 0.0430, 0.0214, 0.0201])
RNN_degree = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
BA_degree = np.array([0.0892,0.3558,1.1754,1.5914,1.7037,1.7502])
Gnp_degree = np.array([1.7115,1.5536,0.5529,0.1433,0.0725,0.0503])

MLP_clustering = np.array([0.0096, 0.0056, 0.0027, 0.0020, 0.0012, 0.0028])
RNN_clustering = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
BA_clustering = np.array([0.0255,0.0881,0.3433,0.4237,0.6041,0.7851])
Gnp_clustering = np.array([0.7683,0.1849,0.1081,0.0146,0.0210,0.0329])


plt.plot(noise,Gnp_degree)
plt.plot(noise,BA_degree)
plt.plot(noise, MLP_degree)
# plt.plot(noise, RNN_degree)

# plt.rc('text', usetex=True)
plt.legend(['E-R','B-A','GraphRNN'])
plt.xlabel('Noise level')
plt.ylabel('MMD degree')

plt.tight_layout()
plt.savefig('figures_paper/robustness_degree.png',dpi=300)
plt.close()

plt.plot(noise,Gnp_clustering)
plt.plot(noise,BA_clustering)
plt.plot(noise, MLP_clustering)
# plt.plot(noise, RNN_clustering)
plt.legend(['E-R','B-A','GraphRNN'])
plt.xlabel('Noise level')
plt.ylabel('MMD clustering')

plt.tight_layout()
plt.savefig('figures_paper/robustness_clustering.png',dpi=300)
plt.close()



