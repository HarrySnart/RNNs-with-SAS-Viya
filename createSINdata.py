import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(2)

T = 20
L = 2000
N = 100

x = np.empty((N, L), 'int64')
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / T).astype('float64')

fig, ax = plt.subplots(1,1,figsize=(10,3))
plt = ax.plot(data[0, :])

SAS.pyplot(fig)

dfout = pd.DataFrame({'y':list(data[0,:])}).reset_index()

SAS.df2sd(dfout,'work.dfout')