import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

np.random.seed(0)

def get_error(h, a, s, sub_tensor):
    h = h.reshape(-1, 3)
    a = a.reshape(-1, 3)
    s = s.reshape(-1, 3)
    ha = np.einsum('Ma, Na -> MNa', h, a)
    hat = np.einsum('MNa, Oa -> MNO', ha, s)
#     print(hat.shape)
    errors = {}
    
    for i in range(a.shape[0]):
        errors[i] = np.sqrt(mean_squared_error(hat[num_train:][i], sub_tensor[num_train:][i]))
    return errors

def get_uncertainty(home_factor, app_factor, A, C, lambda_1, lambda_2):
    var_matrix = np.zeros((len(home_factor), len(app_factor)))
    A_inv = {}
    C_inv = {}
    for i in range(len(A)):
        A_inv[i] = np.linalg.inv(A[i])
    for i in range(len(C)):
        C_inv[i] = np.linalg.inv(C[i])
    
    for i in range(len(home_factor)):
        for j in range(len(app_factor)):
            var_home = np.sqrt(np.matmul(np.matmul(home_factor[i].transpose(), A_inv[i]), home_factor[i]))
            var_app = np.sqrt(np.matmul(np.matmul(app_factor[j].transpose(), C_inv[j]), app_factor[j]))
            var_matrix[i][j] = var_home * lambda_1 + var_app * lambda_2
    return var_matrix

def get_tensor(data, year):
    data_year = data[year]
    homes = data_year.keys()
    tensor = np.zeros((len(homes), num_appliance, 12))
    for idx, id in enumerate(homes):
        tensor[idx] = data_year[id].values.transpose()
    return tensor, homes

all = np.load("../data/data-2013-2017-observed-filtered.npy").item()
year = 2015
num_appliance = 7
lambda_1 = 0.1
lambda_2 = 0.1
lambda_3 = 0.1
appliance = {'use', 'air1', 'clotheswasher1', 'dishwasher1', 'furnace1', 'microwave1', 'refrigerator1'}
tensor, homes = get_tensor(all, year)
num_homes = len(homes)
num_applinace = len(appliance)
num_latent = 3
num_train = 80

A = np.empty([num_homes, num_latent, num_latent])
b = np.empty([num_homes, num_latent, 1])
h = np.empty([num_homes, num_latent, 1])
for i in range(num_homes):
    A[i] = np.identity(num_latent)
    b[i] = np.zeros(3).reshape(-1, 1)
    h[i] = np.random.rand(3).reshape(-1, 1)

C = np.empty([num_appliance, num_latent, num_latent])
d = np.empty([num_appliance, num_latent, 1])
a = np.empty([num_appliance, num_latent, 1])
for i in range(num_appliance):
    C[i] = np.identity(num_latent)
    d[i] = np.zeros(3).reshape(-1, 1)
    a[i] = np.random.rand(3).reshape(-1, 1)

observed_index = {}
observed_index['rows'] = np.asarray(range(num_homes))
observed_index['columns'] = np.zeros(num_homes, dtype=int)
mask_matrix = np.ones((num_homes, num_appliance), dtype=int)
mask_matrix[:, 0] = 0


E = np.empty([12, num_latent, num_latent])
f = np.empty([12, num_latent, 1])
s = np.empty([12, num_latent, 1])
k = 20
errors = {}
for t in range(12):
    E[t] = np.identity(num_latent)
    f[t] = np.zeros(3).reshape(-1, 1)
    s[t] = np.random.rand(3).reshape(-1, 1)
    # get the matrix of current month
    sub_tensor = tensor[:, :, t]
#     print(len(observed_index['rows']))
    for i in range(len(observed_index['rows'])):
        hid = observed_index['rows'][i]
        aid = observed_index['columns'][i]
#         print(hid, aid, sub_tensor[hid][aid])

        A[hid] += np.matmul(np.multiply(a[aid], s[t]), np.multiply(a[aid], s[t]).transpose())
        b[hid] += sub_tensor[hid][aid] * (np.multiply(a[aid], s[t]))
        h[hid] = np.matmul(np.linalg.inv(A[hid]), b[hid])
        
        C[aid] += np.matmul(np.multiply(h[hid], s[t]), np.multiply(h[hid], s[t]).transpose())
        d[aid] += sub_tensor[hid][aid] * (np.multiply(h[hid], s[t]))
        a[aid] = np.matmul(np.linalg.inv(C[aid]), d[aid])
        
        E[t] += np.matmul(np.multiply(a[aid], h[hid]), np.multiply(a[aid], h[hid]).transpose())
        f[t] += sub_tensor[hid][aid] * (np.multiply(a[aid], h[hid]))
        s[t] = np.matmul(np.linalg.inv(E[t]), f[t])
        
    # calculate the prediction error
    errors[t] = get_error(h, a, s[t], sub_tensor)
#     print(error)
    
    # select the (home, appliance) pair to acquire data
    uncertainty = get_uncertainty(h, a, A, C, 0.1, 0.1)
    uncertainty = np.multiply(mask_matrix, uncertainty)
    home_index, app_index = np.unravel_index(np.argsort(uncertainty[:num_train].ravel())[-k:], uncertainty[:num_train].shape)
    for i in range(len(home_index)):
        mask_matrix[home_index[i]][app_index[i]] = 0
    
    
    observed_index['rows'] = np.append(observed_index['rows'], home_index)
    observed_index['columns'] = np.append(observed_index['columns'], app_index)

print(errors)