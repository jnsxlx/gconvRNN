import numpy as np
# np.random.seed(1)
import matplotlib.pyplot as plt
import pickle

from statsmodels.tsa.vector_ar.var_model import VARProcess

''' 
Sample time series data by VAR model
'''

# (ndarray (p x k x k)) – coefficients for lags of endog, part or params reshaped
coefs = np.array([[[0.6, 0, 0, 0.1, 0, -0.2, 0, 0, 0.1, 0], # node 1
                   [0, 0.7, 0, 0, 0, 0, 0, 0, 0, 0],       # node 2
                   [0, 0, 0.7, 0, 0.2, 0, 0, 0, 0, 0],     # node 3
                   [0.1, 0, 0, 0.6, 0, 0.2, 0, 0, 0.2, 0], # node 4
                   [0, 0, 0.2, 0, 0.7, 0, 0, 0, 0, 0],     # node 5
                   [-0.2, 0, 0, 0.2, 0, 0.6, 0, 0, 0.1, 0], # node 6
                   [0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0],       # node 7
                   [0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0],       # node 8
                   [0.1, 0, 0, 0.2, 0, 0.1, 0, 0, 0.6, 0], # node 9
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7]        # node 10
                   ]])

# (ndarray (k x k)) – residual covariance
coefs_exog = np.array([-1, 2, -3, 4, -5, 6, -7, 8, -9, 10]) # (ndarray) – parameters for trend and user provided exog
sigma_u = np.array([[1, 0, 0, 0.5, 0, 0.5, 0, 0, 0.5, 0], # node 1
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # node 2
                    [0, 0, 1, 0, 0.5, 0, 0, 0, 0, 0],  # node 3
                    [0.5, 0, 0, 1, 0, 0.5, 0, 0, 0.5, 0],  # node 4
                    [0, 0, 0.5, 0, 1, 0, 0, 0, 0, 0],  # node 5
                    [0.5, 0, 0, 0.5, 0, 1, 0, 0, 0.5, 0],  # node 6
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # node 7
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # node 8
                    [0.5, 0, 0, 0.5, 0, 0.5, 0, 0, 1, 0],  # node 9
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # node 10
                    ])

# intercepts are 0s if coefs_exog=None
model = VARProcess(coefs=coefs, coefs_exog=coefs_exog, sigma_u=sigma_u)

# check whether the VAR process is stable and output long run intercept of stable VAR process
print(model)

# check whether the VAR process is stable and output its eigenvalues
# print(model.is_stable(verbose=True))

# simulate VAR data, steps=1000 if none
simulated_VARdata = model.simulate_var(steps=505000, seed=1)
data_train = simulated_VARdata[5000:405000, ].tolist()
data_test = simulated_VARdata[405000:455000, ].tolist()
data_valid = simulated_VARdata[455000:505000, ].tolist()


# plot VAR data
plt.plot(simulated_VARdata)
plt.savefig('figs/VAR_simulated_data.eps', transparent=True)
plt.savefig('figs/VAR_simulated_data.png', transparent=True)
plt.show()

# store simulated data to .txt files
# with open('./datasets/simulated_ts/data_train.txt', 'w') as output:
#     for item in data_train:
#         output.write("%s\n" % item)

# # store simulated data to disk
# VARdata_train_pickleout = open("./datasets/simulated_ts/data_train.pickle", "wb")
# pickle.dump(data_train, VARdata_train_pickleout)
# VARdata_train_pickleout.close()
#
# VARdata_test_pickleout = open("./datasets/simulated_ts/data_test.pickle", "wb")
# pickle.dump(data_test, VARdata_test_pickleout)
# VARdata_test_pickleout.close()
#
# VARdata_valid_pickleout = open("./datasets/simulated_ts/data_valid.pickle", "wb")
# pickle.dump(data_test, VARdata_valid_pickleout)
# VARdata_valid_pickleout.close()
