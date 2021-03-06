import numpy as np
import sys
sys.path.append("../code/")
from structure import *
from algo_fix_season import *
from basic import *
import pandas as pd
import matplotlib.pyplot as plt
from autograd.numpy import linalg as LA
import pickle


def active_sensing(year=2015, method='active', latent_dimension=3, alpha1=0.1, alpha2=0.1, alpha3=0.1, lambda_=1, init='random', uncertainty='one', k=5, normalization='none', random_seed=0, iters=5):
    AS = {}
    app_matrix = {}
    season_matrix = {}
    selected_pair = {}    
    
    # for each fold, do the active learning
    for fold_num in range(5):
        tensor, homeids = get_tensor(year, 'missing')
        print("Fold: ", fold_num)
        # train, test = get_train_test(tensor, fold_num=fold_num)
        train, test, tr_homes, tt_homes = get_train_test(tensor, homeids, fold_num=fold_num)
        init_list = None
        app = None
        season = None
        if init == 'pre':
            # season factors are learned with 10 instrumented homes from previou year.
            inst_tensor, init_list = get_instrumented_data(year, 'missing', tr_homes, 10, random_seed=random_seed)
            h, app, season = factorization(inst_tensor, num_latent=latent_dimension, dis=True, random_seed=random_seed)
            

        AS[fold_num] = ActiveSensing_fix(train_tensor=train, test_tensor=test, init_list=init_list, pre_app=None, pre_season=season, 
                                    method=method, latent_dimension=latent_dimension, alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, 
                                    lambda_=lambda_, init=init, uncertainty=uncertainty, k=k, 
                                    normalization=normalization, random_seed=random_seed)
        for t in range(12):
            AS[fold_num].select(t)
            # learn home and appliance factor from random values
            AS[fold_num].update_ALS_random(t, iters)
            print("In iteration {}, Observed items: ".format(t), AS[fold_num].ob_list_length[t])
        app_matrix[fold_num] = AS[fold_num].app_matrix
        season_matrix[fold_num] = AS[fold_num].season_matrix
        selected_pair[fold_num] = AS[fold_num].selected_pair
    
    # get the prediction
    pred = {}
    for t in range(12):
        pred[t] = np.empty((tensor.shape))
        num_home = 0
        for fold_num in range(5):
            pr = np.einsum('Hr, Ar, Sr -> HAS', AS[fold_num].test_home_matrix[t], 
                           AS[fold_num].app_matrix[t], AS[fold_num].season_matrix[t])
            pred[t][num_home:(num_home + AS[fold_num].test_home_matrix[t].shape[0])] = pr
            num_home += AS[fold_num].test_home_matrix[t].shape[0]
    mask_tensor = ~np.isnan(tensor)
    
    # get the accumulated RMSE for each month
    error = {}
    for t in range(12):
        error[t] = np.sqrt(mean_squared_error(pred[t][:, :, :(t+1)][mask_tensor[:, :, :(t+1)]], tensor[:, :, :(t+1)][mask_tensor[:, :, :(t+1)]]))
    
    # get RMSE for each month
    month_error = {}
    for t in range(12):
        month_error[t] = {}
        for i in range(t, 12):
            month_error[t][i] = np.sqrt(mean_squared_error(pred[i][:, :, t][mask_tensor[:, :, t]], tensor[:, :, t][mask_tensor[:, :, t]]))
    # get RMSE for each applianc in each month
    app_error = {}
    for a in range(1, 9):
        app_error[a] = {}
        for t in range(12):
            app_error[a][t] = {}
            for i in range(t, 12):
                app_error[a][t][i] = np.sqrt(mean_squared_error(pred[i][:, a, t][mask_tensor[:, a, t]], tensor[:, a, t][mask_tensor[:, a, t]]))
    
    return error, month_error, app_error, app_matrix, season_matrix, selected_pair

# tensor = get_tensor(2015, 'filter')
year, method, init, normalization, uncertainty, alpha1, alpha2, alpha3, k, latent_dimension, random_seed = sys.argv[1:]

lambda_ = 10000.0
iters = 5
# if init == 'pre':
#     iters = 1  
year = int(year)
alpha1 = float(alpha1)
alpha2 = float(alpha2)
alpha3 = float(alpha3)
random_seed = int(random_seed)
latent_dimension = int(latent_dimension)
k = int(k)
# print("Select {} pairs".format(k))
# error, m_error, app_error, app_matrix, season_matrix, selected_pair = get_error(method=method, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, init=init, normalization=normalization, iters=iters)
# train, test = get_train_test(tensor, fold_num=)

error, month_error, app_error, app_matrix, season_matrix, selected_pair = active_sensing(year=year, method=method, 
                                                                                          latent_dimension=latent_dimension, 
                                                                                          alpha1=alpha1, 
                                                                                          alpha2=alpha2, 
                                                                                          alpha3=alpha3, 
                                                                                          lambda_=lambda_, 
                                                                                          init=init, 
                                                                                          uncertainty=uncertainty,
                                                                                          k=k, 
                                                                                          normalization=normalization, 
                                                                                          random_seed=random_seed, iters=iters)
m_error = pd.DataFrame(month_error).values
a_error = {}
for a in range(1, 9):
    a_error[a] = pd.DataFrame(app_error[a]).values

directory = "../data/fixed_inst/{}/".format(year)
e_file = "error-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(method, init, normalization, uncertainty, alpha1, alpha2, alpha3, k, latent_dimension, random_seed)

m_file = "m-err-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(method, init, normalization, uncertainty, alpha1, alpha2, alpha3, k, latent_dimension,random_seed)
a_file = "a-err-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(method, init, normalization, uncertainty, alpha1, alpha2, alpha3, k, latent_dimension,random_seed)
sp_file = "sp-err-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(method, init, normalization, uncertainty, alpha1, alpha2, alpha3, k,latent_dimension, random_seed)

am_file = "app-matrix-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(method, init, normalization, uncertainty, alpha1, alpha2, alpha3, k, latent_dimension,random_seed)
sm_file = "season-matrix-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(method, init, normalization, uncertainty, alpha1, alpha2, alpha3, k, latent_dimension,random_seed)

f = open(directory+am_file, "wb")
pickle.dump(app_matrix, f)
f.close()
f = open(directory+sm_file, "wb")
pickle.dump(season_matrix, f)
f.close()

f = open(directory+e_file, "wb")
pickle.dump(error, f)
f.close()
f = open(directory+m_file, "wb")
pickle.dump(m_error, f)
f.close()
f = open(directory+a_file, "wb")
pickle.dump(app_error, f)
f.close()
f = open(directory+sp_file, "wb")
pickle.dump(selected_pair, f)
f.close()