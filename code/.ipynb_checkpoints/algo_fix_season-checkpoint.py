from structure import *
# from basic import get_appliance_order
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel
from scipy import stats
import sys
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


class ActiveSensing_fix:
    def __init__(self, train_tensor, test_tensor, init_list=None, pre_app=None, pre_season=None, reg=False, method='active',
                 latent_dimension=3, alpha1=0.1, alpha2=0.1, alpha3=0.1, lambda_=1, init='random', uncertainty='one',
                 k=5, normalization='none', random_seed=0):

        np.random.seed(random_seed)
        self.random_seed = random_seed
        self.latent_dimension = latent_dimension
        self.num_train = train_tensor.shape[0]
        self.num_test = test_tensor.shape[0]
        self.num_home = self.num_train + self.num_test
        self.num_appliance = train_tensor.shape[1]
        self.num_season = train_tensor.shape[2]
        self.pre_app = pre_app
        self.pre_season = pre_season
        self.reg = reg
        # self.appliance_order = ['use', 'air1', 'clotheswasher1', 'drye1', 'dishwasher1', 'furnace1', 'kitchenapp1',
        #                         'microwave1', 'refrigerator1']

        if method != 'active' and normalization != 'none':
            print(">>>>EXIT: only active learning needs normalization!")
            sys.exit(0)

        if method == 'active_2' and uncertainty != 'one' and uncertainty != 'normalized_one' and uncertainty != 'normalized_one_2':
            print(">>>>EXIT: uncertainty reduction only for current uncertinty!")
            sys.exit(0)

        if init == 'pre':
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

        # initialize the factors
        self.homes = []
        for i in range(self.num_home):
            self.homes.append(ActiveSensingHomeStruct(i, self.latent_dimension, lambda_, init, reg, random_seed=random_seed))

        self.apps = []
        for i in range(self.num_appliance):
            if pre_app is not None:
                self.apps.append(ActiveSensingAppStruct(i, self.latent_dimension, lambda_, init, reg, pre_app[i],
                                                        random_seed=random_seed))
            else:
                self.apps.append(ActiveSensingAppStruct(i, self.latent_dimension, lambda_, init, reg,
                                                        random_seed=random_seed))

        self.seasons = []
        for i in range(self.num_season):
            if pre_season is not None:
                self.seasons.append(ActiveSensingSeasonStruct(i, self.latent_dimension, lambda_, init, reg, pre_season[i],
                                                              random_seed=random_seed))
            else:
                self.seasons.append(ActiveSensingSeasonStruct(i, self.latent_dimension, lambda_, init, reg,
                                                              random_seed=random_seed))

        self.init = init
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.method = method
        self.init_list = init_list
        self.k = k
        self.lambda_ = lambda_
        self.train_tensor = train_tensor
        self.test_tensor = test_tensor
        self.normalization = normalization
        self.ob_list_length = {}
        for t in range(12):
            self.ob_list_length[t] = 0
        self.uncertainty_method = uncertainty
        self.ob_list = {}
        self.mask_matrix = np.zeros((self.num_train, self.num_appliance), dtype=bool)
        # if init_list != None:
        #     self.mask_matrix[init_list] = 0

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
        self.uncertainty_detail = {}
        self.future_uncertainty = {}
        self.future_uncertainty_detail = {}
        self.second_order_uncertainty = {}
        self.train_errors = {}
        self.test_errors = {}
        self.test_relative_errors = {}
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
                if self.mask_matrix[i, j] == True and ~np.isnan(self.train_tensor[i, j, season_id]):
                    self.ob_list[self.ob_list_length[season_id]] = [i, j, season_id, self.train_tensor[i, j, season_id]]
                    self.ob_list_length[season_id] += 1

        # update the observed list with the test set: only aggregate
        for i in range(self.num_test):
            if ~np.isnan(self.test_tensor[i, 0, season_id]):
                self.ob_list[self.ob_list_length[season_id]] = [i + self.num_train, 0, season_id, self.test_tensor[i, 0, season_id]]
                self.ob_list_length[season_id] += 1

    def update_second_uncertainty(self, season_id):
        self.future_uncertainty[season_id] = np.zeros((self.num_train, self.num_appliance))
        self.future_uncertainty_detail[season_id] = {}
        self.future_uncertainty_detail[season_id]['home'] = np.zeros((self.num_train, self.num_appliance))
        self.future_uncertainty_detail[season_id]['app'] = np.zeros((self.num_train, self.num_appliance))
        self.future_uncertainty_detail[season_id]['season'] = np.zeros((self.num_train, self.num_appliance))

        if self.uncertainty_method == "one":
            for i in range(self.num_train):
                for j in range(self.num_appliance):
                    # first update A and C matrix by pretending to get the energy readings based on current info
                    self.homes[i].fake_update_parameters(self.apps[j], self.seasons[season_id + 1])
                    self.apps[j].fake_update_parameters(self.homes[i], self.seasons[season_id + 1])
                    # based on the fake A and C to calculte the uncertainty
                    h_circle_a = np.multiply(self.homes[i].h, self.apps[j].a)
                    h_circle_s = np.multiply(self.homes[i].h, self.seasons[season_id + 1].s)
                    a_circle_s = np.multiply(self.apps[j].a, self.seasons[season_id + 1].s)

                    var_1 = np.sqrt(np.matmul(np.matmul(a_circle_s.transpose(), self.homes[i].fakeAInv), a_circle_s))
                    var_2 = np.sqrt(np.matmul(np.matmul(h_circle_s.transpose(), self.apps[j].fakeCInv), h_circle_s))
                    var_3 = np.sqrt(np.matmul(np.matmul(h_circle_a.transpose(), self.seasons[season_id].EInv), h_circle_a))

                    self.future_uncertainty[season_id][i, j] = var_1 * self.alpha1 + var_2 * self.alpha2 + var_3 * self.alpha3
                    self.future_uncertainty_detail[season_id]['home'][i, j] = var_1
                    self.future_uncertainty_detail[season_id]['app'][i, j] = var_2
                    self.future_uncertainty_detail[season_id]['season'][i, j] = var_3

        if self.uncertainty_method == "normalized_one":
            for i in range(self.num_train):
                for j in range(self.num_appliance):
                    # first update A and C matrix by pretending to get the energy readings based on current info
                    self.homes[i].fake_update_parameters(self.apps[j], self.seasons[season_id + 1])
                    self.apps[j].fake_update_parameters(self.homes[i], self.seasons[season_id + 1])
                    # based on the fake A and C to calculte the uncertainty
                    h_circle_a = np.multiply(self.homes[i].h, self.apps[j].a)
                    h_circle_s = np.multiply(self.homes[i].h, self.seasons[season_id + 1].s)
                    a_circle_s = np.multiply(self.apps[j].a, self.seasons[season_id + 1].s)

                    var_1 = np.sqrt(np.matmul(np.matmul(a_circle_s.transpose(), self.homes[i].fakeAInv), a_circle_s)) / np.linalg.norm(self.apps[j].a)
                    var_2 = np.sqrt(np.matmul(np.matmul(h_circle_s.transpose(), self.apps[j].fakeCInv), h_circle_s)) / np.linalg.norm(self.homes[i].h)
                    var_3 = np.sqrt(np.matmul(np.matmul(h_circle_a.transpose(), self.seasons[season_id].EInv), h_circle_a))

                    self.future_uncertainty[season_id][i, j] = var_1 * self.alpha1 + var_2 * self.alpha2 + var_3 * self.alpha3
                    self.future_uncertainty_detail[season_id]['home'][i, j] = var_1
                    self.future_uncertainty_detail[season_id]['app'][i, j] = var_2
                    self.future_uncertainty_detail[season_id]['season'][i, j] = var_3

        if self.uncertainty_method == "normalized_one_2":
            for i in range(self.num_train):
                for j in range(self.num_appliance):
                    # first update A and C matrix by pretending to get the energy readings based on current info
                    self.homes[i].fake_update_parameters(self.apps[j], self.seasons[season_id + 1])
                    self.apps[j].fake_update_parameters(self.homes[i], self.seasons[season_id + 1])
                    # based on the fake A and C to calculte the uncertainty
                    h_circle_a = np.multiply(self.homes[i].h, self.apps[j].a)
                    h_circle_s = np.multiply(self.homes[i].h, self.seasons[season_id + 1].s)
                    a_circle_s = np.multiply(self.apps[j].a, self.seasons[season_id + 1].s)

                    var_1 = np.sqrt(np.matmul(np.matmul(a_circle_s.transpose(), self.homes[i].fakeAInv), a_circle_s)) / np.linalg.norm(self.homes[i].h)
                    var_2 = np.sqrt(np.matmul(np.matmul(h_circle_s.transpose(), self.apps[j].fakeCInv), h_circle_s)) / np.linalg.norm(self.homes[i].h)
                    var_3 = np.sqrt(np.matmul(np.matmul(h_circle_a.transpose(), self.seasons[season_id].EInv), h_circle_a))

                    self.future_uncertainty[season_id][i, j] = var_1 * self.alpha1 + var_2 * self.alpha2 + var_3 * self.alpha3
                    self.future_uncertainty_detail[season_id]['home'][i, j] = var_1
                    self.future_uncertainty_detail[season_id]['app'][i, j] = var_2
                    self.future_uncertainty_detail[season_id]['season'][i, j] = var_3

    def get_uncertainty(self, season_id, home_id):
        ucty = np.ones(self.num_appliance)
        for j in range(self.num_appliance):
            h_circle_a = np.multiply(self.homes[home_id].h, self.apps[j].a)
            h_circle_s = np.multiply(self.homes[home_id].h, self.seasons[season_id].s)
            a_circle_s = np.multiply(self.apps[j].a, self.seasons[season_id].s)

            var_1 = np.sqrt(np.matmul(np.matmul(a_circle_s.transpose(), self.homes[home_id].AInv), a_circle_s))
            var_2 = np.sqrt(np.matmul(np.matmul(h_circle_s.transpose(), self.apps[j].CInv), h_circle_s))
            var_3 = np.sqrt(np.matmul(np.matmul(h_circle_a.transpose(), self.seasons[season_id].EInv), h_circle_a))
            ucty[j] = var_1 * self.alpha1 + var_2 * self.alpha2 + var_3 * self.alpha3
        return ucty

    def get_future_uncertainty(self, season_id, home_id):

        future_ucty = np.ones(self.num_appliance)

        for j in range(self.num_appliance):
            # first update A and C matrix by pretending to get the energy readings based on current info
            self.homes[home_id].fake_update_parameters(self.apps[j], self.seasons[season_id + 1])
            self.apps[j].fake_update_parameters(self.homes[home_id], self.seasons[season_id + 1])
            self.seasons[season_id].fake_update_parameters(self.homes[home_id], self.apps[j])
            # based on the fake A and C to calculte the uncertainty
            h_circle_a = np.multiply(self.homes[home_id].h, self.apps[j].a)
            h_circle_s = np.multiply(self.homes[home_id].h, self.seasons[season_id + 1].s)
            a_circle_s = np.multiply(self.apps[j].a, self.seasons[season_id + 1].s)

            var_1 = np.sqrt(np.matmul(np.matmul(a_circle_s.transpose(), self.homes[home_id].fakeAInv), a_circle_s))
            var_2 = np.sqrt(np.matmul(np.matmul(h_circle_s.transpose(), self.apps[j].fakeCInv), h_circle_s))
            var_3 = np.sqrt(np.matmul(np.matmul(h_circle_a.transpose(), self.seasons[season_id].fakeEInv), h_circle_a))
            future_ucty[j] = var_1 * self.alpha1 + var_2 * self.alpha2 + var_3 * self.alpha3

        return future_ucty

    def update_uncertainty(self, season_id):
        self.uncertainty[season_id] = np.zeros((self.num_train, self.num_appliance))
        self.uncertainty_detail[season_id] = {}
        self.uncertainty_detail[season_id]['home'] = np.zeros((self.num_train, self.num_appliance))
        self.uncertainty_detail[season_id]['app'] = np.zeros((self.num_train, self.num_appliance))
        self.uncertainty_detail[season_id]['season'] = np.zeros((self.num_train, self.num_appliance))

        if self.uncertainty_method == 'one':
            # print("uncertainty is one")
            for i in range(self.num_train):
                for j in range(self.num_appliance):
                    h_circle_a = np.multiply(self.homes[i].h, self.apps[j].a)
                    h_circle_s = np.multiply(self.homes[i].h, self.seasons[season_id].s)
                    a_circle_s = np.multiply(self.apps[j].a, self.seasons[season_id].s)

                    var_1 = np.sqrt(np.matmul(np.matmul(a_circle_s.transpose(), self.homes[i].AInv), a_circle_s))
                    var_2 = np.sqrt(np.matmul(np.matmul(h_circle_s.transpose(), self.apps[j].CInv), h_circle_s))
                    var_3 = np.sqrt(np.matmul(np.matmul(h_circle_a.transpose(), self.seasons[season_id].EInv), h_circle_a))

                    self.uncertainty[season_id][i, j] = var_1 * self.alpha1 + var_2 * self.alpha2 + var_3 * self.alpha3
                    self.uncertainty_detail[season_id]['home'][i, j] = var_1
                    self.uncertainty_detail[season_id]['app'][i, j] = var_2
                    self.uncertainty_detail[season_id]['season'][i, j] = var_3

        if self.uncertainty_method == 'agg':
            # print("uncertainty is one")
            for i in range(self.num_train):
                for j in range(self.num_appliance):
                    h_circle_a = np.multiply(self.homes[i].h, self.apps[j].a)
                    h_circle_s = np.multiply(self.homes[i].h, self.seasons[season_id].s)
                    a_circle_s = np.multiply(self.apps[j].a, self.seasons[season_id].s)

                    var_1 = np.sqrt(np.matmul(np.matmul(a_circle_s.transpose(), self.homes[i].AInv), a_circle_s))
                    var_2 = np.sqrt(np.matmul(np.matmul(h_circle_s.transpose(), self.apps[j].CInv), h_circle_s))
                    var_3 = np.sqrt(np.matmul(np.matmul(h_circle_a.transpose(), self.seasons[season_id].EInv), h_circle_a))

                    self.uncertainty[season_id][i, j] = var_1 * self.alpha1 + var_2 * self.alpha2 + var_3 * self.alpha3
                    self.uncertainty_detail[season_id]['home'][i, j] = var_1
                    self.uncertainty_detail[season_id]['app'][i, j] = var_2
                    self.uncertainty_detail[season_id]['season'][i, j] = var_3
            agg_uncertainty = self.uncertainty[season_id][i, 0]
            for j in range(self.num_appliance):
                self.uncertainty[season_id][:, j] += agg_uncertainty

        if self.uncertainty_method == 'normalized_one':
            for i in range(self.num_train):
                for j in range(self.num_appliance):
                    h_circle_a = np.multiply(self.homes[i].h, self.apps[j].a)
                    h_circle_s = np.multiply(self.homes[i].h, self.seasons[season_id].s)
                    a_circle_s = np.multiply(self.apps[j].a, self.seasons[season_id].s)

                    var_1 = np.sqrt(np.matmul(np.matmul(a_circle_s.transpose(), self.homes[i].AInv), a_circle_s)) / np.linalg.norm(self.apps[j].a)
                    var_2 = np.sqrt(np.matmul(np.matmul(h_circle_s.transpose(), self.apps[j].CInv), h_circle_s)) / np.linalg.norm(self.homes[i].h)
                    var_3 = np.sqrt(np.matmul(np.matmul(h_circle_a.transpose(), self.seasons[season_id].EInv), h_circle_a))

                    self.uncertainty[season_id][i, j] = var_1 * self.alpha1 + var_2 * self.alpha2 + var_3 * self.alpha3
                    self.uncertainty_detail[season_id]['home'][i, j] = var_1
                    self.uncertainty_detail[season_id]['app'][i, j] = var_2
                    self.uncertainty_detail[season_id]['season'][i, j] = var_3

        if self.uncertainty_method == 'normalized_one_2':
            for i in range(self.num_train):
                for j in range(self.num_appliance):
                    h_circle_a = np.multiply(self.homes[i].h, self.apps[j].a)
                    h_circle_s = np.multiply(self.homes[i].h, self.seasons[season_id].s)
                    a_circle_s = np.multiply(self.apps[j].a, self.seasons[season_id].s)

                    var_1 = np.sqrt(np.matmul(np.matmul(a_circle_s.transpose(), self.homes[i].AInv), a_circle_s)) / np.linalg.norm(self.homes[i].h)
                    var_2 = np.sqrt(np.matmul(np.matmul(h_circle_s.transpose(), self.apps[j].CInv), h_circle_s)) / np.linalg.norm(self.homes[i].h)
                    var_3 = np.sqrt(np.matmul(np.matmul(h_circle_a.transpose(), self.seasons[season_id].EInv), h_circle_a))

                    self.uncertainty[season_id][i, j] = var_1 * self.alpha1 + var_2 * self.alpha2 + var_3 * self.alpha3
                    self.uncertainty_detail[season_id]['home'][i, j] = var_1
                    self.uncertainty_detail[season_id]['app'][i, j] = var_2
                    self.uncertainty_detail[season_id]['season'][i, j] = var_3
                    # print(var_1, var_2, var_3)
        if self.uncertainty_method == 'next':
            for i in range(self.num_train):
                for j in range(self.num_appliance):
                    h_circle_a = np.multiply(self.homes[i].h, self.apps[j].a)
                    h_circle_s = np.multiply(self.homes[i].h, self.seasons[season_id + 1].s)
                    a_circle_s = np.multiply(self.apps[j].a, self.seasons[season_id + 1].s)

                    var_1 = np.sqrt(np.matmul(np.matmul(a_circle_s.transpose(), self.homes[i].AInv), a_circle_s))
                    var_2 = np.sqrt(np.matmul(np.matmul(h_circle_s.transpose(), self.apps[j].CInv), h_circle_s))
                    var_3 = np.sqrt(np.matmul(np.matmul(h_circle_a.transpose(), self.seasons[season_id].EInv), h_circle_a))

                    self.uncertainty[season_id][i, j] = var_1 * self.alpha1 + var_2 * self.alpha2 + var_3 * self.alpha3
                    self.uncertainty_detail[season_id]['home'][i, j] = var_1
                    self.uncertainty_detail[season_id]['app'][i, j] = var_2
                    self.uncertainty_detail[season_id]['season'][i, j] = var_3

        if self.uncertainty_method == 'equal':

            num_future_month = self.num_season - season_id - 1
            if num_future_month != 0:
                weight = 0.67 / num_future_month
            else:
                weight = 0

            for t in range(season_id, self.num_season):
                for i in range(self.num_train):
                    for j in range(self.num_appliance):
                        h_circle_a = np.multiply(self.homes[i].h, self.apps[j].a)
                        h_circle_s = np.multiply(self.homes[i].h, self.seasons[t].s)
                        a_circle_s = np.multiply(self.apps[j].a, self.seasons[t].s)

                        var_1 = np.sqrt(np.matmul(np.matmul(a_circle_s.transpose(), self.homes[i].AInv), a_circle_s))
                        var_2 = np.sqrt(np.matmul(np.matmul(h_circle_s.transpose(), self.apps[j].CInv), h_circle_s))
                        var_3 = np.sqrt(np.matmul(np.matmul(h_circle_a.transpose(), self.seasons[season_id].EInv), h_circle_a))

                        uncty = var_1 * self.alpha1 + var_2 * self.alpha2 + var_3 * self.alpha3
                        if t == season_id:
                            self.uncertainty[season_id][i, j] += 0.33 * uncty
                        else:
                            self.uncertainty[season_id][i, j] += weight * uncty

                        self.uncertainty_detail[season_id]['home'][i, j] = var_1
                        self.uncertainty_detail[season_id]['app'][i, j] = var_2
                        self.uncertainty_detail[season_id]['season'][i, j] = var_3

        if self.uncertainty_method == 'weighted':
            for t in range(season_id, self.num_season):
                for i in range(self.num_train):
                    for j in range(self.num_appliance):
                        h_circle_a = np.multiply(self.homes[i].h, self.apps[j].a)
                        h_circle_s = np.multiply(self.homes[i].h, self.seasons[t].s)
                        a_circle_s = np.multiply(self.apps[j].a, self.seasons[t].s)

                        var_1 = np.sqrt(np.matmul(np.matmul(a_circle_s.transpose(), self.homes[i].AInv), a_circle_s))
                        var_2 = np.sqrt(np.matmul(np.matmul(h_circle_s.transpose(), self.apps[j].CInv), h_circle_s))
                        var_3 = np.sqrt(np.matmul(np.matmul(h_circle_a.transpose(), self.seasons[season_id].EInv), h_circle_a))

                        # self.uncertainty[season_id][i, j] += (var_1 * self.alpha1 + var_2 * self.alpha2 + var_3 * self.alpha3) / (t + 1 - season_id)
                        self.uncertainty[season_id][i, j] += (var_1 * self.alpha1 + var_2 * self.alpha2 + var_3 * self.alpha3) / pow(3, t + 1 - season_id)
                        self.uncertainty_detail[season_id]['home'][i, j] = var_1
                        self.uncertainty_detail[season_id]['app'][i, j] = var_2
                        self.uncertainty_detail[season_id]['season'][i, j] = var_3

        if self.normalization == 'zscore':
            self.uncertainty[season_id] = np.abs(stats.zscore(self.uncertainty[season_id], axis=0))
        # self.uncertainty[season_id] *= 100
        # print(self.uncertainty[season_id])

    def get_covariance_matrix(self, season_id):
        # for the covariance matrix of home factors
        cor_home = rbf_kernel(self.train_home_matrix[season_id])
        cor_appliance = rbf_kernel(self.app_matrix[season_id])
        return cor_home, cor_appliance

    def get_mutual_info(self, season_id):
        # for home factors
        home_matrix = self.train_home_matrix[season_id]
        d_home = np.zeros(self.num_train)
        index = np.arange(self.num_train)
        for i in range(self.num_train):
            delta_i = rbf_kernel(home_matrix[i].reshape(1, -1))
            new_index = np.delete(index, i)
            remain_matrix = home_matrix[new_index]
            sigma_iUi = rbf_kernel(home_matrix[i].reshape(1, -1), remain_matrix)
            sigma_UiUi = np.linalg.inv(rbf_kernel(remain_matrix, remain_matrix))
            sigma_Uii = rbf_kernel(remain_matrix, home_matrix[i].reshape(1, -1))
            delta_iUi = delta_i - np.dot(sigma_iUi, np.dot(sigma_UiUi, sigma_Uii))
            print(delta_iUi)
            d_home[i] = 0.5 * np.log(delta_i / delta_iUi)

        # app_matrix = self.app_matrix[season_id]
        # d_app = np.zeros(self.num_appliance)
        # index = np.arange(self.num_appliance)
        # for i in range(self.num_appliance):
        #     delta_i  = rbf_kernel(app_matrix[i].reshape(1, -1))
        #     new_index = np.delete(index, i)
        #     remain_matrix = app_matrix[new_index]
        #     sigma_iUi = rbf_kernel(app_matrix[i].reshape(1, -1), remain_matrix)
        #     sigma_UiUi = np.linalg.inv(rbf_kernel(remain_matrix, remain_matrix))
        #     sigma_Uii = rbf_kernel(remain_matrix, app_matrix[i].reshape(1, -1))
        #     delta_iUi = delta_i - np.dot(sigma_iUi, np.dot(sigma_UiUi, sigma_Uii))
        #     d_app[i] = 0.5 * np.log(delta_i / delta_iUi)

        return d_home

    def select(self, season_id):
        # for the first month, add all aggregate readings as observed
        if self.method == 'all':
            self.mask_matrix[:] = True

        else:
            # if season_id == 0 and self.init == 'random':
            self.selected_pair[season_id] = {}
            if season_id == 0:
                self.mask_matrix[:, 0] = True
                # observed homes at the initialization
                if self.init_list is not None:
                    # ob_homes = np.random.choice(self.num_train, self.init_ob)
                    self.mask_matrix[self.init_list, :] = True

            else:
                self.selected_pair[season_id] = {}
                mask_array = self.mask_matrix.flatten()
                if self.method == 'random':
                    np.random.seed(self.random_seed)
                    random_candidate = []
                    for i, val in enumerate(mask_array):
                        if val == False:
                            random_candidate.append(i)
                    index = np.random.choice(random_candidate, self.k)
                    home_index, appliance_index = divmod(index, self.num_appliance)

                if self.method == 'active':
                    # get the uncertainty of the factors learned for last season
                    self.update_uncertainty(season_id - 1)
                    # uncertainty = np.multiply(self.mask_matrix, self.uncertainty[season_id-1])
                    uncertainty = self.uncertainty[season_id - 1].copy()
                    uncertainty[self.mask_matrix] = -np.inf
                    home_index, appliance_index = np.unravel_index(np.argsort(uncertainty.ravel())[-self.k:],
                                                                   uncertainty.shape)

                if self.method == 'active_2':
                    # get the uncertainty of the factors learned for last season
                    self.update_uncertainty(season_id - 1)
                    # update the future uncertainty based on current information
                    self.update_second_uncertainty(season_id - 1)
                    # remove the ones acquired before
                    self.second_order_uncertainty[season_id - 1] = self.uncertainty[season_id - 1] - self.future_uncertainty[season_id - 1]
                    uncertainty_reduction = np.abs(self.second_order_uncertainty[season_id - 1])
                    uncertainty_reduction[self.mask_matrix] = -np.inf
                    home_index, appliance_index = np.unravel_index(np.argsort(uncertainty_reduction.ravel())[-self.k:],
                                                                   uncertainty_reduction.shape)

                if self.method == 'active_error':
                    # get the uncertainty of the factors learned for last month
                    self.update_uncertainty(season_id - 1)
                    # globally normalize the uncertainty
                    normalized_uncty = self.uncertainty[season_id - 1] / self.uncertainty[season_id - 1].max()
                    # get aggregate errors for each home
                    # k = self.get_aggregate_error(season_id - 1)
                    # print(k.shape)
                    agg_error = np.repeat(self.get_aggregate_error(season_id - 1).reshape(-1, 1), self.num_appliance, axis=1)
                    normalized_agg_error = agg_error / agg_error.max()

                    score = 0.9 * normalized_uncty + 0.1 * normalized_agg_error

                    score[self.mask_matrix] = -np.inf
                    home_index, appliance_index = np.unravel_index(np.argsort(score.ravel())[-self.k:],
                                                                   score.shape)

                if self.method == 'oracle':
                    # get the error of prediction
                    pred = np.einsum('Hr, Ar, Sr -> HAS',
                                     self.train_home_matrix[season_id - 1], self.app_matrix[season_id - 1],
                                     self.season_matrix[season_id])
                    absolute_error = np.abs(pred[:, :, season_id] - self.train_tensor[:, :, season_id])
                    new_error = np.nan_to_num(np.multiply(self.mask_matrix, absolute_error))
                    home_index, appliance_index = np.unravel_index(np.argsort(new_error.ravel())[-self.k:],
                                                                   new_error.shape)

                if self.method == 'cluster':
                    # clustering the homes
                    np.random.seed(self.random_seed)
                    kmeans = KMeans(n_clusters=self.k, random_state=self.random_seed).fit(self.train_home_matrix[season_id - 1])
                    home_index, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, self.train_home_matrix[season_id - 1])
                    appliance_index = []

                    for i, homeid in enumerate(home_index):
                        uncertainty = self.get_uncertainty(season_id - 1, homeid)
                        uncertainty[self.mask_matrix[homeid]] = -np.inf
                        appliance_index.append(np.random.choice(np.flatnonzero(uncertainty == uncertainty.max())))

                if self.method == 'cluster_2':
                    np.random.seed(self.random_seed)
                    kmeans = KMeans(n_clusters=self.k, random_state=self.random_seed).fit(self.train_home_matrix[season_id - 1])
                    home_index, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, self.train_home_matrix[season_id - 1])
                    appliance_index = []
                    for i, homeid in enumerate(home_index):
                        uncertainty = self.get_uncertainty(season_id - 1, homeid)
                        future_uncertainty = self.get_future_uncertainty(season_id - 1, homeid)
                        uncertainty_reduction = np.abs(uncertainty - future_uncertainty)
                        uncertainty_reduction[self.mask_matrix[homeid]] = -np.inf
                        appliance_index.append(np.random.choice(np.flatnonzero(uncertainty_reduction == uncertainty_reduction.max())))

                if self.method == 'cluster_3':
                    kmeans = KMeans(n_clusters=self.k, random_state=self.random_seed).fit(self.train_home_matrix[season_id - 1])

                    self.update_uncertainty(season_id - 1)
                    # uncertainty = np.multiply(self.mask_matrix, self.uncertainty[season_id-1])
                    uncertainty = self.uncertainty[season_id - 1]
                    uncertainty[self.mask_matrix] = -np.inf

                    home_index = []
                    appliance_index = []
                    for i in range(self.k):
                        tmp_uncertainty = uncertainty.copy()
                        tmp_uncertainty[kmeans.labels_ != i, :] = -np.inf
                        hid, aid = np.unravel_index(np.argsort(tmp_uncertainty.ravel())[-self.k:],
                                                    tmp_uncertainty.shape)
                        home_index.append(hid[0])
                        appliance_index.append(aid[0])

                if self.method == 'cluster_4':
                    kmeans = KMeans(n_clusters=self.k, random_state=self.random_seed).fit(self.train_home_matrix[season_id - 1])

                    self.update_uncertainty(season_id - 1)
                    self.update_second_uncertainty(season_id - 1)
                    self.second_order_uncertainty[season_id - 1] = self.uncertainty[season_id - 1] - self.future_uncertainty[season_id - 1]
                    uncertainty_reduction = np.abs(self.second_order_uncertainty[season_id - 1])

                    # uncertainty = np.multiply(self.mask_matrix, self.uncertainty[season_id-1])
                    uncertainty_reduction[self.mask_matrix] = -np.inf

                    home_index = []
                    appliance_index = []
                    for i in range(self.k):
                        tmp_uncertainty_reduction = uncertainty_reduction.copy()
                        tmp_uncertainty_reduction[kmeans.labels_ != i, :] = -np.inf
                        hid, aid = np.unravel_index(np.argsort(tmp_uncertainty_reduction.ravel())[-self.k:],
                                                    tmp_uncertainty_reduction.shape)
                        home_index.append(hid[0])
                        appliance_index.append(aid[0])

                if self.method == 'mutualinfo':
                    # print("Select mutualinfo")
                    # get the corvariance matrix of home factors
                    # get the corvariance matrix of appliance factors
                    # cor_home, cor_app = get_covariance_matrix(season_id - 1)
                    d_home = self.get_mutual_info(season_id - 1)
                    home_index = d_home.argsort()[-self.k:][::-1]
                    # candidate_app = d_app.argsort()[::-1]
                    appliance_index = np.random.randint(1, self.num_appliance, self.k)
                    # get the mutual information of homes
                    # get the mutual information of appliances
                    # select the (home, appliance) pairs

                for i in range(len(home_index)):
                    self.mask_matrix[home_index[i], appliance_index[i]] = True
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

    def update_als_update_season(self, season_id, num_iterations=10):

        # Randomly init home factors
        self.homes = []
        for i in range(self.num_home):
            self.homes.append(ActiveSensingHomeStruct(i, self.latent_dimension, self.lambda_, self.init, self.reg,
                                                      random_seed=self.random_seed))
        # Randomly init appliance factors
        self.apps = []
        for i in range(self.num_appliance):
            if self.pre_app is not None and season_id == 0:
                self.apps.append(ActiveSensingAppStruct(i, self.latent_dimension, self.lambda_, self.init, self.reg,
                                                        self.pre_app[i], random_seed=self.random_seed))
            else:
                self.apps.append(ActiveSensingAppStruct(i, self.latent_dimension, self.lambda_, self.init, self.reg,
                                                        random_seed=self.random_seed))
        # init season factors with previous learned factors if pre_season is not None
        self.seasons = []
        for i in range(self.num_season):
            if self.pre_season is not None:
                self.seasons.append(ActiveSensingSeasonStruct(i, self.latent_dimension, self.lambda_, self.init, self.reg,
                                                              self.pre_season[i], random_seed=self.random_seed))
            else:
                self.seasons.append(ActiveSensingSeasonStruct(i, self.latent_dimension, self.lambda_, self.init, self.reg,
                                                              random_seed=self.random_seed))

        # update the factors based on the observations
        ob_list = self.ob_list.copy()
        np.random.seed(self.random_seed)
        for iters in range(num_iterations):
            # randomly shuffle the observations
            np.random.shuffle(ob_list)
            # first update the home factors
            for i in ob_list:
                hid, aid, sid, e = ob_list[i]
                self.homes[hid].update_parameters(self.apps[aid], self.seasons[sid], e)
            # then update the appliance factors
            for i in ob_list:
                hid, aid, sid, e = ob_list[i]
                self.apps[aid].update_parameters(self.homes[hid], self.seasons[sid], e)
            # update the season factors (only the E matrix)
            for i in ob_list:
                hid, aid, sid, e = ob_list[i]
                self.seasons[sid].update_parameters(self.homes[hid], self.apps[aid], e)
        # update corresponding matrices and the errors
        self.update_matrix(season_id)
        self.update_train_error(season_id)
        self.update_test_error(season_id)

    def update_als_fix_season(self, season_id, num_iterations=10):

        # Randomly init home factors
        self.homes = []
        for i in range(self.num_home):
            self.homes.append(ActiveSensingHomeStruct(i, self.latent_dimension, self.lambda_, self.init, self.reg,
                                                      random_seed=self.random_seed))
        # Randomly init appliance factors
        self.apps = []
        for i in range(self.num_appliance):
            if self.pre_app is not None and season_id == 0:
                self.apps.append(ActiveSensingAppStruct(i, self.latent_dimension, self.lambda_, self.init, self.reg,
                                                        self.pre_app[i], random_seed=self.random_seed))
            else:
                self.apps.append(ActiveSensingAppStruct(i, self.latent_dimension, self.lambda_, self.init, self.reg,
                                                        random_seed=self.random_seed))
        # init season factors with previous learned factors if pre_season is not None
        self.seasons = []
        for i in range(self.num_season):
            if self.pre_season is not None:
                self.seasons.append(ActiveSensingSeasonStruct(i, self.latent_dimension, self.lambda_, self.init, self.reg,
                                                              self.pre_season[i], random_seed=self.random_seed))
            else:
                self.seasons.append(ActiveSensingSeasonStruct(i, self.latent_dimension, self.lambda_, self.init, self.reg,
                                                              random_seed=self.random_seed))

        # update the factors based on the observations
        ob_list = self.ob_list.copy()
        np.random.seed(self.random_seed)
        for iters in range(num_iterations):
            # randomly shuffle the observations
            np.random.shuffle(ob_list)
            # first update the home factors
            for i in ob_list:
                hid, aid, sid, e = ob_list[i]
                self.homes[hid].update_parameters(self.apps[aid], self.seasons[sid], e)
            # then update the appliance factors
            for i in ob_list:
                hid, aid, sid, e = ob_list[i]
                self.apps[aid].update_parameters(self.homes[hid], self.seasons[sid], e)
            # update the season factors (only the E matrix)
            for i in ob_list:
                hid, aid, sid, e = ob_list[i]
                self.seasons[sid].update_parameters_2(self.homes[hid], self.apps[aid], e)
        # update corresponding matrices and the errors
        self.update_matrix(season_id)
        self.update_train_error(season_id)
        self.update_test_error(season_id)

    def update_als_update(self, season_id, num_iterations=10):

        # update the home and appliance factors based on the observations
        ob_list = self.ob_list.copy()
        np.random.seed(self.random_seed)
        for iters in range(num_iterations):
            # randomly shuffle the observations
            np.random.shuffle(ob_list)
            # first update the home factors
            for i in ob_list:
                hid, aid, sid, e = ob_list[i]
                self.homes[hid].update_parameters(self.apps[aid], self.seasons[sid], e)
            # the update the appliance factors
            for i in ob_list:
                hid, aid, sid, e = ob_list[i]
                self.apps[aid].update_parameters(self.homes[hid], self.seasons[sid], e)
        # update corresponding matrices and the errors
        self.update_matrix(season_id)
        self.update_train_error(season_id)
        self.update_test_error(season_id)

    def update_train_error(self, season_id):
        mask_tensor = ~np.isnan(self.train_tensor)
        pred = np.einsum('Hr, Ar, Sr -> HAS',
                         self.train_home_matrix[season_id], self.app_matrix[season_id], self.season_matrix[season_id])

        # get overall error
        errors = {}
        for i in range(self.num_appliance):
            errors[i] = np.sqrt(mean_squared_error(pred[:, i, :(season_id + 1)][mask_tensor[:, i, :(season_id + 1)]],
                                                   self.train_tensor[:, i, :(season_id + 1)][mask_tensor[:, i, :(season_id + 1)]]))
        self.train_errors[season_id] = errors

        # get error for each month
        for i in range(season_id + 1):
            month_error = {}
            for j in range(self.num_appliance):
                month_error[j] = np.sqrt(mean_squared_error(pred[:, j, i][mask_tensor[:, j, i]],
                                                            self.train_tensor[:, j, i][mask_tensor[:, j, i]]))
            self.train_errors_per_month[i][season_id] = month_error

    def update_test_error(self, season_id):
        mask_tensor = ~np.isnan(self.test_tensor)
        # HA = np.einsum('Hr, Ar -> HAr', self.test_home_matrix, self.app_matrix)
        pred = np.einsum('Hr, Ar, Sr -> HAS',
                         self.test_home_matrix[season_id], self.app_matrix[season_id], self.season_matrix[season_id])

        errors = {}
        for i in range(1, self.num_appliance):
            errors[i] = np.sqrt(mean_squared_error(pred[:, i, :(season_id + 1)][mask_tensor[:, i, :(season_id + 1)]],
                                                   self.test_tensor[:, i, :(season_id + 1)][mask_tensor[:, i, :(season_id + 1)]]))
        self.test_errors[season_id] = errors

        for i in range(season_id + 1):
            month_error = {}
            for j in range(self.num_appliance):
                month_error[j] = np.sqrt(mean_squared_error(pred[:, j, i][mask_tensor[:, j, i]],
                                                            self.test_tensor[:, j, i][mask_tensor[:, j, i]]))
            self.test_errors_per_month[i][season_id] = month_error

    def get_relative_aggregate_error(self, season_id):
        mask_tensor = ~np.isnan(self.train_tensor)
        # HA = np.einsum('Hr, Ar -> HAr', self.test_home_matrix, self.app_matrix)
        pred = np.einsum('Hr, Ar, Sr -> HAS',
                         self.train_home_matrix[season_id], self.app_matrix[season_id], self.season_matrix[season_id])
        
        diff_matrix = pred[:, 0, :(season_id + 1)][mask_tensor[:, 0, :(season_id + 1)]] / self.train_tensor[:, 0, :(season_id + 1)][mask_tensor[:, 0, :(season_id + 1)]] - 1

        diff_matrix = diff_matrix.reshape(-1, season_id + 1)
        mse = (diff_matrix ** 2).mean(axis=1)
        return mse.reshape(-1, 1)

    def get_aggregate_error(self, season_id):
        # mask_tensor = ~np.isnan(self.train_tensor)
        # HA = np.einsum('Hr, Ar -> HAr', self.test_home_matrix, self.app_matrix)
        pred = np.einsum('Hr, Ar, Sr -> HAS',
                         self.train_home_matrix[season_id], self.app_matrix[season_id], self.season_matrix[season_id])

        # diff_matrix = np.abs(pred[:, 0, :(season_id + 1)] - self.train_tensor[:, 0, :(season_id + 1)]).
        diff_matrix = np.abs(pred[:, 0, season_id] - self.train_tensor[:, 0, season_id])

        return diff_matrix

    def get_aggregate_uncertainty(self, season_id):
        # print("uncertainty is one")
        uncty = np.zeros((self.num_train, season_id + 1))
        for i in range(self.num_train):
            for j in range(season_id + 1):
                h_circle_a = np.multiply(self.homes[i].h, self.apps[0].a)
                h_circle_s = np.multiply(self.homes[i].h, self.seasons[j].s)
                a_circle_s = np.multiply(self.apps[0].a, self.seasons[j].s)

                var_1 = np.sqrt(np.matmul(np.matmul(a_circle_s.transpose(), self.homes[i].AInv), a_circle_s))
                var_2 = np.sqrt(np.matmul(np.matmul(h_circle_s.transpose(), self.apps[0].CInv), h_circle_s))
                var_3 = np.sqrt(np.matmul(np.matmul(h_circle_a.transpose(), self.seasons[j].EInv), h_circle_a))

                uncty[i, j] = var_1 * self.alpha1 + var_2 * self.alpha2 + var_3 * self.alpha3
        return uncty

    def display_test_error(self, season_id, app=True):
        print('Month:', season_id)
        if app:
            for i in range(self.num_appliance):
                print(self.appliance_order[i], self.test_errors[season_id][i])
        else:
            print(pd.Series(self.test_errors[season_id]).mean())
        print('*' * 20)

    def display_train_error(self, season_id, app=True):
        print('Month:', season_id)
        if app:
            for i in range(self.num_appliance):
                print(self.appliance_order[i], self.train_errors[season_id][i])
        else:
            print(pd.Series(self.train_errors[season_id]).mean())
        print('*' * 20)
