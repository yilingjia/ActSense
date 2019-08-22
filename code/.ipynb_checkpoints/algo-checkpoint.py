import numpy as np
from structure import *
from sklearn.metrics import mean_squared_error
from scipy import stats
import sys
from sys import exit
import pandas as pd

class ActiveSensing:
    def __init__(self, train_tensor, test_tensor, init_list=None, pre_app=None, pre_season=None, method='active', latent_dimension=3, 
        alpha1=0.1, alpha2=0.1, alpha3=0.1, lambda_=1, init='random', uncertainty='one', 
        k=10, normalization='none', random_seed=0):
        
        np.random.seed(random_seed)
        self.latent_dimension = latent_dimension
        self.num_train = train_tensor.shape[0]
        self.num_test = test_tensor.shape[0]
        self.num_home = self.num_train + self.num_test
        self.num_appliance = train_tensor.shape[1]
        self.num_season = train_tensor.shape[2]

        self.appliance_order = ['use', 'air1', 'clotheswasher1', 'drye1', 'dishwasher1', 'furnace1', 'kitchenapp1','microwave1','refrigerator1']


        if method != 'active' and normalization != 'none':
            print(">>>>EXIT: only active learning needs normalization!")
            sys.exit(0)
        if init=='pre':
            if pre_app is None and pre_season is None:
                print(">>>>EXIT: Need initialized appliance factor or season factor!")
                sys.exit(0)
            if pre_app is not None and pre_app[0].shape[0] != latent_dimension:
                print(">>>>EXIT: Dimension of pre_app is not equal to latent_dimension!")
                sys.exit(0)
            if pre_season is not None and pre_season[0].shape[0] != latent_dimension:
                print(">>>>EXIT: Dimension of pre_season is not equal to latent_dimension!")
                sys.exit(0)
        else:
            if uncertainty != 'one':
                print(">>>>EXIT: Equal or weighted uncertainty is only for previous initialization!")
                sys.exit(0)


        self.homes = []
        for i in range(self.num_home):
            self.homes.append(ActiveSensingHomeStruct(i, self.latent_dimension, lambda_, init, random_seed=random_seed))
        self.apps = []
        for i in range(self.num_appliance):
            if pre_app is not None:
                self.apps.append(ActiveSensingAppStruct(i, self.latent_dimension, lambda_, init, pre_app[i], random_seed=random_seed))
            else:
                self.apps.append(ActiveSensingAppStruct(i, self.latent_dimension, lambda_, init, random_seed=random_seed))
        self.seasons = []
        for i in range(self.num_season):
            if pre_season is not None:
                self.seasons.append(ActiveSensingSeasonStruct(i, self.latent_dimension, lambda_, init, pre_season[i], random_seed=random_seed))
            else:
                self.seasons.append(ActiveSensingSeasonStruct(i, self.latent_dimension, lambda_, init, random_seed=random_seed))
        
        self.init = init
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.method = method
        self.init_list = init_list
        self.k = k
        self.train_tensor = train_tensor
        self.test_tensor = test_tensor
        self.normalization = normalization
        self.ob_list_length = {}
        for t in range(12):
            self.ob_list_length[t] = 0
        self.uncertainty_method = uncertainty
        self.ob_list = {}
        self.mask_matrix = np.ones((self.num_train, self.num_appliance), dtype=int)
        # if init_list != None:
        #     self.mask_matrix[init_list] = 0
        self.train = train_tensor
        self.test = test_tensor
        self.train_home_matrix = {}
        self.test_home_matrix = {}
        self.app_matrix = {}
        self.season_matrix = {}
        for t in range(12):
            self.train_home_matrix[t] = np.zeros((self.num_train, self.latent_dimension))
            self.test_home_matrix[t] = np.zeros((self.num_test, self.latent_dimension))
            self.app_matrix[t] = np.zeros((self.num_appliance, self.latent_dimension))
            self.season_matrix[t] = np.zeros((self.num_season, self.latent_dimension))
        self.uncertainty = {}
        self.train_errors = {}
        self.test_errors = {}
        self.train_errors_per_month = {}
        self.test_errors_per_month = {}
        self.selected_pair = {}
        for i in range(self.num_season):
            self.train_errors_per_month[i] = {}
            self.test_errors_per_month[i] = {}

    def update_observed_list(self, season_id):
        # update the observed list with the training set
        if season_id != 0:
            self.ob_list_length[season_id] = self.ob_list_length[season_id - 1]
        for i in range(self.num_train):
            for j in range(self.num_appliance):
                if self.mask_matrix[i, j] == 0 and ~np.isnan(self.train[i, j, season_id]):
                    self.ob_list[self.ob_list_length[season_id]] = [i, j, season_id, self.train[i, j, season_id]]
                    #if j != 0:
                        #print(self.ob_list[self.ob_list_length[season_id]])
                    self.ob_list_length[season_id]+= 1

        # update the observed list with the test set: only aggregate
        for i in range(self.num_test):
            if ~np.isnan(self.test[i, 0, season_id]):
                self.ob_list[self.ob_list_length[season_id]] = [i + self.num_train, 0, season_id, self.test[i, 0, season_id]]
                self.ob_list_length[season_id] += 1

    def update_uncertainty(self, season_id):
        self.uncertainty[season_id] = np.zeros((self.num_train, self.num_appliance))

        if self.uncertainty_method == 'one':
            for i in range(self.num_train):
                for j in range(self.num_appliance):
                    h_circ_a = np.multiply(self.homes[i].h, self.apps[j].a)
                    h_circ_s = np.multiply(self.homes[i].h, self.seasons[season_id].s)
                    a_circ_s = np.multiply(self.apps[j].a, self.seasons[season_id].s)

                    var_1 = np.sqrt(np.matmul(np.matmul(a_circ_s.transpose(), self.homes[i].AInv), a_circ_s))
                    var_2 = np.sqrt(np.matmul(np.matmul(h_circ_s.transpose(), self.apps[j].CInv), h_circ_s))
                    var_3 = np.sqrt(np.matmul(np.matmul(h_circ_a.transpose(), self.seasons[season_id].EInv), h_circ_a))

                    self.uncertainty[season_id][i, j] = var_1 * self.alpha1 + var_2 * self.alpha2 + var_3 * self.alpha3
        
        if self.uncertainty_method == 'equal':
            for t in range(season_id, self.num_season):
                for i in range(self.num_train):
                    for j in range(self.num_appliance):
                        h_circ_a = np.multiply(self.homes[i].h, self.apps[j].a)
                        h_circ_s = np.multiply(self.homes[i].h, self.seasons[t].s)
                        a_circ_s = np.multiply(self.apps[j].a, self.seasons[t].s)

                        var_1 = np.sqrt(np.matmul(np.matmul(a_circ_s.transpose(), self.homes[i].AInv), a_circ_s))
                        var_2 = np.sqrt(np.matmul(np.matmul(h_circ_s.transpose(), self.apps[j].CInv), h_circ_s))
                        var_3 = np.sqrt(np.matmul(np.matmul(h_circ_a.transpose(), self.seasons[season_id].EInv), h_circ_a))

                        self.uncertainty[season_id][i, j] += var_1 * self.alpha1 + var_2 * self.alpha2 + var_3 * self.alpha3

        if self.uncertainty_method == 'weighted':
            for t in range(season_id, self.num_season):
                for i in range(self.num_train):
                    for j in range(self.num_appliance):
                        h_circ_a = np.multiply(self.homes[i].h, self.apps[j].a)
                        h_circ_s = np.multiply(self.homes[i].h, self.seasons[t].s)
                        a_circ_s = np.multiply(self.apps[j].a, self.seasons[t].s)

                        var_1 = np.sqrt(np.matmul(np.matmul(a_circ_s.transpose(), self.homes[i].AInv), a_circ_s))
                        var_2 = np.sqrt(np.matmul(np.matmul(h_circ_s.transpose(), self.apps[j].CInv), h_circ_s))
                        var_3 = np.sqrt(np.matmul(np.matmul(h_circ_a.transpose(), self.seasons[season_id].EInv), h_circ_a))

                        self.uncertainty[season_id][i, j] += (var_1 * self.alpha1 + var_2 * self.alpha2 + var_3 * self.alpha3)/(t+1-season_id)
 

        if self.normalization == 'zscore':
            self.uncertainty[season_id] = np.abs(stats.zscore(self.uncertainty[season_id], axis=0))

    def select(self, season_id):
        # for the first month, add all aggregate readings as observed
        if self.method == 'all':
            self.mask_matrix[:] = 0

        else:
            # if season_id == 0 and self.init == 'random':
            if season_id == 0:
                self.mask_matrix[:, 0] = 0
                # observed homes at the initialization
                if self.init_list != None:
                    # ob_homes = np.random.choice(self.num_train, self.init_ob)
                    self.mask_matrix[self.init_list, :] = 0

            else:
                self.selected_pair[season_id] = {}
                mask_array = self.mask_matrix.flatten()
                if self.method == 'random':
                    
                    random_candidate = []
                    for i, val in enumerate(mask_array):
                        if val == 1:
                            random_candidate.append(i)
                    index = np.random.choice(random_candidate, self.k)
                    home_index, appliance_index = divmod(index, self.num_appliance)

                if self.method == 'active':
                    # get the uncertainty of the factors learned for last season
                    self.update_uncertainty(season_id-1)
                    uncertainty = np.multiply(self.mask_matrix, self.uncertainty[season_id-1])
                    home_index, appliance_index = np.unravel_index(np.argsort(uncertainty.ravel())[-self.k:], uncertainty.shape)

                for i in range(len(home_index)):
                    self.mask_matrix[home_index[i], appliance_index[i]] = 0
                    self.selected_pair[season_id][i] = [home_index[i], appliance_index[i]]

        self.update_observed_list(season_id)

    def update_matrix(self, season_id):
        for i in range(self.num_train):
            self.train_home_matrix[season_id][i] = self.homes[i].h.reshape(1, self.latent_dimension)
        for i in range(self.num_test):
            self.test_home_matrix[season_id][i] = self.homes[i + self.num_train].h.reshape(1, self.latent_dimension)
        for i in range(self.num_appliance):
            self.app_matrix[season_id][i] = self.apps[i].a.reshape(1, self.latent_dimension)
        for i in range(self.num_season):
            self.season_matrix[season_id][i] = self.seasons[i].s.reshape(1, self.latent_dimension)

    def update_ALS(self, season_id, num_iterations=10):

        # first update with new readings
        if season_id != 0:
            for i in range(self.ob_list_length[season_id-1], self.ob_list_length[season_id]):
                hid, aid, sid, e = self.ob_list[i]
                self.seasons[sid].updateParameters(self.homes[hid], self.apps[aid], e)
                #self.apps[aid].updateParameters(self.homes[hid], self.seasons[sid], e)
                #self.homes[hid].updateParameters(self.apps[aid], self.seasons[sid], e)
                

        # then iteratively update the model with all the observations
        ob_list = self.ob_list.copy()
        for iters in range(num_iterations):
            np.random.shuffle(ob_list)
            for i in ob_list:
                hid, aid, sid, e = ob_list[i]
                
                self.homes[hid].updateParameters(self.apps[aid], self.seasons[sid], e)
                self.apps[aid].updateParameters(self.homes[hid], self.seasons[sid], e)
                self.seasons[sid].updateParameters(self.homes[hid], self.apps[aid], e)
                
        self.update_matrix(season_id)
        self.get_train_error(season_id)
        self.get_test_error(season_id)
    
    def update_ALS_2(self, season_id, num_iterations=10):
        ob_list = self.ob_list.copy()
        for iters in range(num_iterations):
            # first update seasonal factors with the aggregate readings
            np.random.shuffle(ob_list)
            if season_id == 0:
                for i in ob_list:
                    hid, aid, sid, e = ob_list[i]
                    self.homes[hid].updateParameters(self.apps[aid], self.seasons[sid], e)
                for i in ob_list:
                    hid, aid, sid, e = ob_list[i]
                    self.apps[aid].updateParameters(self.homes[hid], self.seasons[sid], e)
                # last use all the readings to update home factors
                for i in ob_list:
                    hid, aid, sid, e = ob_list[i]
                    #if aid == 0:
                    self.seasons[sid].updateParameters(self.homes[hid], self.apps[aid], e)
            else:
                for i in ob_list:
                    hid, aid, sid, e = ob_list[i]
                    #if aid == 0:
                    self.seasons[sid].updateParameters(self.homes[hid], self.apps[aid], e)
                for i in ob_list:
                    hid, aid, sid, e = ob_list[i]
                    self.homes[hid].updateParameters(self.apps[aid], self.seasons[sid], e)
                for i in ob_list:
                    hid, aid, sid, e = ob_list[i]
                    
                    self.apps[aid].updateParameters(self.homes[hid], self.seasons[sid], e)
                    #if aid == 1:
                    #    print(hid, aid, sid, e)
                    #    print(self.apps[aid].a)
                # last use all the readings to update home factors
            #self.get_train_error(season_id)
            #self.display_train_error(season_id, False)
            # then use all readings to update the appliance factors
            
            
        self.update_matrix(season_id)
        self.get_train_error(season_id)
        self.get_test_error(season_id)
           
    def update_ALS_3(self, season_id, num_iterations=10):
        
        self.homes = []
        for i in range(self.num_home):
            self.homes.append(ActiveSensingHomeStruct(i, self.latent_dimension, self.lambda_, self.init, random_seed=self.random_seed))
        self.apps = []
        for i in range(self.num_appliance):
            if self.pre_app is not None:
                self.apps.append(ActiveSensingAppStruct(i, self.latent_dimension, self.lambda_, self.init, self.pre_app[i], random_seed=self.random_seed))
            else:
                self.apps.append(ActiveSensingAppStruct(i, self.latent_dimension, self.lambda_, self.init, random_seed=self.random_seed))
        self.seasons = []
        for i in range(self.num_season):
            if self.pre_season is not None:
                self.seasons.append(ActiveSensingSeasonStruct(i, self.latent_dimension, self.lambda_, self.init, self.pre_season[i], random_seed=self.random_seed))
            else:
                self.seasons.append(ActiveSensingSeasonStruct(i, self.latent_dimension, self.lambda_, self.init, random_seed=self.random_seed))
                
        ob_list = self.ob_list.copy()
        for iters in range(num_iterations):
            # first update seasonal factors with the aggregate readings
            np.random.shuffle(ob_list)
            #print(ob_list[:10])
            for i in ob_list:
                hid, aid, sid, e = ob_list[i]
                #if aid == 0:
                self.seasons[sid].updateParameters(self.homes[hid], self.apps[aid], e)
            for i in ob_list:
                hid, aid, sid, e = ob_list[i]
                self.apps[aid].updateParameters(self.homes[hid], self.seasons[sid], e)  
            for i in ob_list:
                hid, aid, sid, e = ob_list[i]
                self.homes[hid].updateParameters(self.apps[aid], self.seasons[sid], e)
            
            # last use all the readings to update home factors
            
            # then use all readings to update the appliance factors
            
            
        self.update_matrix(season_id)
        self.get_train_error(season_id)
        self.get_test_error(season_id)

    def get_train_error(self, season_id):
        mask_tensor = ~np.isnan(self.train_tensor)
        # HA = np.einsum('Hr, Ar -> HAr', self.train_home_matrix, self.app_matrix)
        HAS = np.einsum('Hr, Ar, Sr -> HAS', self.train_home_matrix[season_id], self.app_matrix[season_id], self.season_matrix[season_id])

        # get overall error
        errors = {}
        for i in range(self.num_appliance):
            errors[i] = np.sqrt(mean_squared_error(HAS[:, i, :(season_id+1)][mask_tensor[:, i, :(season_id+1)]], self.train_tensor[:, i, :(season_id+1)][mask_tensor[:, i, :(season_id+1)]]))
        self.train_errors[season_id] = errors

        # get error for each month
        for i in range(season_id+1):
            month_error = {}
            for j in range(self.num_appliance):
                month_error[j] = np.sqrt(mean_squared_error(HAS[:, j, i][mask_tensor[:, j, i]], self.train_tensor[:, j, i][mask_tensor[:, j, i]]))
            self.train_errors_per_month[i][season_id] = month_error 

    def get_test_error(self, season_id):
        mask_tensor = ~np.isnan(self.test_tensor)
        # HA = np.einsum('Hr, Ar -> HAr', self.test_home_matrix, self.app_matrix)
        HAS = np.einsum('Hr, Ar, Sr -> HAS', self.test_home_matrix[season_id], self.app_matrix[season_id], self.season_matrix[season_id])

        errors = {}
        for i in range(self.num_appliance):
            errors[i] = np.sqrt(mean_squared_error(HAS[:, i, :(season_id+1)][mask_tensor[:, i, :(season_id+1)]], self.test_tensor[:, i, :(season_id+1)][mask_tensor[:, i, :(season_id+1)]]))
        self.test_errors[season_id] = errors

        for i in range(season_id+1):
            month_error = {}
            for j in range(self.num_appliance):
                month_error[j] = np.sqrt(mean_squared_error(HAS[:, j, i][mask_tensor[:, j, i]], self.test_tensor[:, j, i][mask_tensor[:, j, i]]))
            self.test_errors_per_month[i][season_id] = month_error 

    def get_uncertainty(self):
        return self.uncertainty

    def display_test_error(self, season_id, app=True):
        print('Month:', season_id)
        if app:
            for i in range(self.num_appliance):
                print(self.appliance_order[i], self.test_errors[season_id][i])
        else:
            print(pd.Series(self.test_errors[season_id]).mean())
        print('*'*20)

    def display_train_error(self, season_id, app=True):
        print('Month:', season_id)
        if app:
            for i in range(self.num_appliance):
                print(self.appliance_order[i], self.train_errors[season_id][i])
        else:
            print(pd.Series(self.train_errors[season_id]).mean())
        print('*'*20)
