from structure import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel
from scipy import stats
import sys
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import math


class ActiveSensing_fix:
    def __init__(self, train_tensor, test_tensor, init_list=None, pre_app=None, pre_season=None, reg=False, method='active',
                 latent_dimension=3, alpha1=0.1, alpha2=0.1, alpha3=0.1, lambda1=10000, lambda2=10000, lambda3=10000,
                 gamma1=0.5, gamma2=0.5, init='random', uncertainty='current', kernel='gaussian', sigma=12,
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
        self.gamma1 = gamma1
        self.gamma2 = gamma2

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
            self.homes.append(ActiveSensingHomeStruct(i, self.latent_dimension, lambda1, init, reg, random_seed=random_seed))

        self.apps = []
        for i in range(self.num_appliance):
            if pre_app is not None:
                self.apps.append(ActiveSensingAppStruct(i, self.latent_dimension, lambda2, init, reg, pre_app[i],
                                                        random_seed=random_seed))
            else:
                self.apps.append(ActiveSensingAppStruct(i, self.latent_dimension, lambda2, init, reg,
                                                        random_seed=random_seed))

        self.seasons = []
        for i in range(self.num_season):
            if pre_season is not None:
                self.seasons.append(ActiveSensingSeasonStruct(i, self.latent_dimension, lambda3, init, reg, pre_season[i],
                                                              random_seed=random_seed))
            else:
                self.seasons.append(ActiveSensingSeasonStruct(i, self.latent_dimension, lambda3, init, reg,
                                                              random_seed=random_seed))

        self.init = init
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.method = method
        self.init_list = init_list
        self.k = k
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.train_tensor = train_tensor
        self.test_tensor = test_tensor
        self.normalization = normalization
        self.ob_list_length = {}
        for t in range(12):
            self.ob_list_length[t] = 0
        self.uncertainty_method = uncertainty
        self.ob_list = {}
        self.mask_matrix = np.zeros((self.num_train, self.num_appliance), dtype=bool)
        self.aggregate_error = np.zeros((self.num_season, self.num_season, self.num_train))

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
        self.kernel = kernel
        self.sigma = sigma
        for i in range(self.num_season):
            self.train_errors_per_month[i] = {}
            self.test_errors_per_month[i] = {}

    # update the observed list according to the selection.
    def update_observed_list(self, season_id):
        # update the observed list for training set: the selected pairs
        if season_id != 0:
            self.ob_list_length[season_id] = self.ob_list_length[season_id - 1]
        for i in range(self.num_train):
            for j in range(self.num_appliance):
                if self.mask_matrix[i, j] == True and ~np.isnan(self.train_tensor[i, j, season_id]):
                    self.ob_list[self.ob_list_length[season_id]] = [i, j, season_id, self.train_tensor[i, j, season_id]]
                    self.ob_list_length[season_id] += 1

        # update the observed list with the test set: only aggregate readings for all homes
        for i in range(self.num_test):
            if ~np.isnan(self.test_tensor[i, 0, season_id]):
                self.ob_list[self.ob_list_length[season_id]] = [i + self.num_train, 0, season_id, self.test_tensor[i, 0, season_id]]
                self.ob_list_length[season_id] += 1

    # get the model uncertainty: sum of uncertanties from all estimations.
    def get_model_uncertainty(self, season_id):
        model_ucty = np.zeros((self.num_train, self.num_appliance))

        for i in range(self.num_train):
            for j in range(self.num_appliance):
                h_circle_a = np.multiply(self.homes[i].h, self.apps[j].a)
                h_circle_s = np.multiply(self.homes[i].h, self.seasons[season_id].s)
                a_circle_s = np.multiply(self.apps[j].a, self.seasons[season_id].s)

                var_1 = np.sqrt(np.matmul(np.matmul(a_circle_s.transpose(), self.homes[i].AInv), a_circle_s))
                var_2 = np.sqrt(np.matmul(np.matmul(h_circle_s.transpose(), self.apps[j].CInv), h_circle_s))
                var_3 = np.sqrt(np.matmul(np.matmul(h_circle_a.transpose(), self.seasons[season_id].EInv), h_circle_a))

                model_ucty[i, j] = var_1 * self.alpha1 + var_2 * self.alpha2 + var_3 * self.alpha3

        return model_ucty

    # update uncertainty according to the learned latent factors at current time.
    def update_uncertainty(self, season_id):

        self.uncertainty[season_id] = np.zeros((self.num_train, self.num_appliance))
        self.uncertainty_detail[season_id] = {}
        self.uncertainty_detail[season_id]['home'] = np.zeros((self.num_train, self.num_appliance))
        self.uncertainty_detail[season_id]['app'] = np.zeros((self.num_train, self.num_appliance))
        self.uncertainty_detail[season_id]['season'] = np.zeros((self.num_train, self.num_appliance))

        if self.uncertainty_method == 'current':
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

        if self.uncertainty_method == 'previous':
            # print("uncertainty is one")
            for t in range(season_id + 1):
                for i in range(self.num_train):
                    for j in range(self.num_appliance):
                        h_circle_a = np.multiply(self.homes[i].h, self.apps[j].a)
                        h_circle_s = np.multiply(self.homes[i].h, self.seasons[t].s)
                        a_circle_s = np.multiply(self.apps[j].a, self.seasons[t].s)

                        var_1 = np.sqrt(np.matmul(np.matmul(a_circle_s.transpose(), self.homes[i].AInv), a_circle_s))
                        var_2 = np.sqrt(np.matmul(np.matmul(h_circle_s.transpose(), self.apps[j].CInv), h_circle_s))
                        var_3 = np.sqrt(np.matmul(np.matmul(h_circle_a.transpose(), self.seasons[t].EInv), h_circle_a))

                        uncty = var_1 * self.alpha1 + var_2 * self.alpha2 + var_3 * self.alpha3

                        self.uncertainty[season_id][i, j] += uncty
                        self.uncertainty_detail[season_id]['home'][i, j] = var_1
                        self.uncertainty_detail[season_id]['app'][i, j] = var_2
                        self.uncertainty_detail[season_id]['season'][i, j] = var_3

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

        if self.uncertainty_method == 'prev_equal':

            num_future_month = self.num_season - season_id - 1
            num_prev_month = self.num_season - num_future_month

            # obs_uncty = 0
            # future_uncty = 0
            for t in range(self.num_season):
                for i in range(self.num_train):
                    for j in range(self.num_appliance):
                        h_circle_a = np.multiply(self.homes[i].h, self.apps[j].a)
                        h_circle_s = np.multiply(self.homes[i].h, self.seasons[t].s)
                        a_circle_s = np.multiply(self.apps[j].a, self.seasons[t].s)

                        var_1 = np.sqrt(np.matmul(np.matmul(a_circle_s.transpose(), self.homes[i].AInv), a_circle_s))
                        var_2 = np.sqrt(np.matmul(np.matmul(h_circle_s.transpose(), self.apps[j].CInv), h_circle_s))
                        var_3 = np.sqrt(np.matmul(np.matmul(h_circle_a.transpose(), self.seasons[t].EInv), h_circle_a))

                        uncty = var_1 * self.alpha1 + var_2 * self.alpha2 + var_3 * self.alpha3
                        if t <= season_id:
                            self.uncertainty[season_id][i, j] += uncty * self.gamma1 / num_prev_month
                        else:
                            self.uncertainty[season_id][i, j] += uncty * (1 - self.gamma1) / num_future_month

                        self.uncertainty_detail[season_id]['home'][i, j] = var_1
                        self.uncertainty_detail[season_id]['app'][i, j] = var_2
                        self.uncertainty_detail[season_id]['season'][i, j] = var_3

        if self.uncertainty_method == 'prev_future_weighted':

            num_future_month = self.num_season - season_id - 1
            num_prev_month = self.num_season - num_future_month

            for t in range(self.num_season):
                for i in range(self.num_train):
                    for j in range(self.num_appliance):
                        h_circle_a = np.multiply(self.homes[i].h, self.apps[j].a)
                        h_circle_s = np.multiply(self.homes[i].h, self.seasons[t].s)
                        a_circle_s = np.multiply(self.apps[j].a, self.seasons[t].s)

                        var_1 = np.sqrt(np.matmul(np.matmul(a_circle_s.transpose(), self.homes[i].AInv), a_circle_s))
                        var_2 = np.sqrt(np.matmul(np.matmul(h_circle_s.transpose(), self.apps[j].CInv), h_circle_s))
                        if t < season_id:
                            var_3 = np.sqrt(np.matmul(np.matmul(h_circle_a.transpose(), self.seasons[t].EInv), h_circle_a))
                        else:
                            var_3 = np.sqrt(np.matmul(np.matmul(h_circle_a.transpose(), self.seasons[season_id].EInv), h_circle_a))

                        uncty = var_1 * self.alpha1 + var_2 * self.alpha2 + var_3 * self.alpha3
                        distance = np.abs(season_id - t)

                        weight = self.proximity_kernel_function(distance)

                        self.uncertainty[season_id][i, j] += uncty * weight

                        self.uncertainty_detail[season_id]['home'][i, j] = var_1
                        self.uncertainty_detail[season_id]['app'][i, j] = var_2
                        self.uncertainty_detail[season_id]['season'][i, j] = var_3

        if self.uncertainty_method == 'future_weighted':
            num_future_month = self.num_season - season_id - 1
            num_prev_month = self.num_season - num_future_month

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
                        distance = np.abs(season_id - t)

                        weight = self.proximity_kernel_function(distance)

                        self.uncertainty[season_id][i, j] += uncty * weight

                        self.uncertainty_detail[season_id]['home'][i, j] = var_1
                        self.uncertainty_detail[season_id]['app'][i, j] = var_2
                        self.uncertainty_detail[season_id]['season'][i, j] = var_3

        if self.uncertainty_method == 'equal':

            num_future_month = self.num_season - season_id - 1
            if num_future_month != 0:
                weight = (1 - self.gamma1) / num_future_month
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
                            self.uncertainty[season_id][i, j] += self.gamma1 * uncty
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
                        uncty = var_1 * self.alpha1 + var_2 * self.alpha2 + var_3 * self.alpha3
                        if t == season_id:
                            self.uncertainty[season_id][i, j] += self.gamma1 * uncty
                        else:
                            self.uncertainty[season_id][i, j] += uncty * (1 - self.gamma1) / pow(2, t - season_id)

                        self.uncertainty_detail[season_id]['home'][i, j] = var_1
                        self.uncertainty_detail[season_id]['app'][i, j] = var_2
                        self.uncertainty_detail[season_id]['season'][i, j] = var_3

        if self.normalization == 'zscore':
            self.uncertainty[season_id] = np.abs(stats.zscore(self.uncertainty[season_id], axis=0))
        # self.uncertainty[season_id] *= 100
        # print(self.uncertainty[season_id])

    # kernel function for the weight of uncertainties.
    def proximity_kernel_function(self, dist):
        if self.kernel == 'gaussian':
            return np.exp(- (dist ** 2) / (2 * (self.sigma ** 2)))
        if self.kernel == 'triangle':
            if dist > self.sigma:
                return 0
            else:
                return 1 - dist / self.sigma
        if self.kernel == 'cosine':
            if dist > self.sigma:
                return 0
            else:
                return 0.5 * (1 + np.cos(dist * math.pi / self.sigma))
        if self.kernel == 'circle':
            if dist > self.sigma:
                return 0
            else:
                return np.sqrt(1 - (dist / self.sigma) ** 2)

    # select the <home, appliance> pairs according to the assigned method.
    def select(self, season_id):
        # for the first month, add all aggregate readings as observed
        if self.method == 'all':
            self.mask_matrix[:] = True

        else:
            self.selected_pair[season_id] = {}
            # for the first month, randomly select k pairs.
            if season_id == 0:
                self.mask_matrix[:, 0] = True
                mask_array = self.mask_matrix.flatten()
                np.random.seed(self.random_seed)
                random_candidate = []
                for i, val in enumerate(mask_array):
                    if val == False:
                        random_candidate.append(i)
                index = np.random.choice(random_candidate, self.k)
                home_index, appliance_index = divmod(index, self.num_appliance)

                for i in range(len(home_index)):
                    self.mask_matrix[home_index[i], appliance_index[i]] = True
                    self.selected_pair[season_id][i] = [home_index[i], appliance_index[i]]
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
                    uncertainty = self.uncertainty[season_id - 1].copy()
                    uncertainty[self.mask_matrix] = -np.inf
                    home_index, appliance_index = np.unravel_index(np.argsort(uncertainty.ravel())[-self.k:],
                                                                   uncertainty.shape)

                if self.method == 'oracle':
                    # get the error of prediction
                    pred = np.einsum('Hr, Ar, Sr -> HAS',
                                     self.train_home_matrix[season_id - 1], self.app_matrix[season_id - 1],
                                     self.season_matrix[season_id])
                    absolute_error = np.abs(pred[:, :, season_id] - self.train_tensor[:, :, season_id])
                    new_error = np.nan_to_num(np.multiply(self.mask_matrix, absolute_error))
                    home_index, appliance_index = np.unravel_index(np.argsort(new_error.ravel())[-self.k:],
                                                                   new_error.shape)

                for i in range(len(home_index)):
                    self.mask_matrix[home_index[i], appliance_index[i]] = True
                    self.selected_pair[season_id][i] = [home_index[i], appliance_index[i]]

        self.update_observed_list(season_id)

    # update the latent matrices with current parameters
    def update_matrix(self, season_id):
        for i in range(self.num_train):
            self.train_home_matrix[season_id][i] = self.homes[i].h.reshape(1, self.latent_dimension)
        for i in range(self.num_test):
            self.test_home_matrix[season_id][i] = self.homes[i + self.num_train].h.reshape(1, self.latent_dimension)
        for i in range(self.num_appliance):
            self.app_matrix[season_id][i] = self.apps[i].a.reshape(1, self.latent_dimension)
        for i in range(self.num_season):
            self.season_matrix[season_id][i] = self.seasons[i].s.reshape(1, self.latent_dimension)

    # update the parameters with updated season factors
    def update_als_update_season(self, season_id, num_iterations=10):

        # Randomly init home factors
        self.homes = []
        for i in range(self.num_home):
            self.homes.append(ActiveSensingHomeStruct(i, self.latent_dimension, self.lambda1, self.init, self.reg,
                                                      random_seed=self.random_seed))
        # Randomly init appliance factors
        self.apps = []
        for i in range(self.num_appliance):
            if self.pre_app is not None and season_id == 0:
                self.apps.append(ActiveSensingAppStruct(i, self.latent_dimension, self.lambda2, self.init, self.reg,
                                                        self.pre_app[i], random_seed=self.random_seed))
            else:
                self.apps.append(ActiveSensingAppStruct(i, self.latent_dimension, self.lambda2, self.init, self.reg,
                                                        random_seed=self.random_seed))
        # init season factors with previous learned factors if pre_season is not None
        self.seasons = []
        for i in range(self.num_season):
            if self.pre_season is not None:
                self.seasons.append(ActiveSensingSeasonStruct(i, self.latent_dimension, self.lambda3, self.init, self.reg,
                                                              self.pre_season[i], random_seed=self.random_seed))
            else:
                self.seasons.append(ActiveSensingSeasonStruct(i, self.latent_dimension, self.lambda3, self.init, self.reg,
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
        self.update_aggregate_error(season_id)

    # update the parameters with fixed season factor
    def update_als_fix_season(self, season_id, num_iterations=10):

        # Randomly init home factors
        self.homes = []
        for i in range(self.num_home):
            self.homes.append(ActiveSensingHomeStruct(i, self.latent_dimension, self.lambda1, self.init, self.reg,
                                                      random_seed=self.random_seed))
        # Randomly init appliance factors
        self.apps = []
        for i in range(self.num_appliance):
            if self.pre_app is not None and season_id == 0:
                self.apps.append(ActiveSensingAppStruct(i, self.latent_dimension, self.lambda2, self.init, self.reg,
                                                        self.pre_app[i], random_seed=self.random_seed))
            else:
                self.apps.append(ActiveSensingAppStruct(i, self.latent_dimension, self.lambda2, self.init, self.reg,
                                                        random_seed=self.random_seed))
        # init season factors with previous learned factors if pre_season is not None
        self.seasons = []
        for i in range(self.num_season):
            if self.pre_season is not None:
                self.seasons.append(ActiveSensingSeasonStruct(i, self.latent_dimension, self.lambda3, self.init, self.reg,
                                                              self.pre_season[i], random_seed=self.random_seed))
            else:
                self.seasons.append(ActiveSensingSeasonStruct(i, self.latent_dimension, self.lambda3, self.init, self.reg,
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

    # def update_als_update(self, season_id, num_iterations=10):

    #     # update the home and appliance factors based on the observations
    #     ob_list = self.ob_list.copy()
    #     np.random.seed(self.random_seed)
    #     for iters in range(num_iterations):
    #         # randomly shuffle the observations
    #         np.random.shuffle(ob_list)
    #         # first update the home factors
    #         for i in ob_list:
    #             hid, aid, sid, e = ob_list[i]
    #             self.homes[hid].update_parameters(self.apps[aid], self.seasons[sid], e)
    #         # the update the appliance factors
    #         for i in ob_list:
    #             hid, aid, sid, e = ob_list[i]
    #             self.apps[aid].update_parameters(self.homes[hid], self.seasons[sid], e)
    #     # update corresponding matrices and the errors
    #     self.update_matrix(season_id)
    #     self.update_train_error(season_id)
    #     self.update_test_error(season_id)

    # update the training error
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

    # update the test error
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

    def update_aggregate_error(self, season_id):

        pred = np.einsum('Hr, Ar, Sr -> HAS', self.train_home_matrix[season_id], self.app_matrix[season_id], self.season_matrix[season_id])
        pred_agg = pred[:, 0, :(season_id + 1)]
        diff_matrix = np.abs(pred_agg - self.train_tensor[:, 0, :(season_id + 1)])
        self.aggregate_error[season_id, :(season_id + 1)] = diff_matrix.T

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

    def get_appliance_error(self, season_id):
        pred = np.einsum('Hr, Ar, Sr -> HAS',
                         self.train_home_matrix[season_id], self.app_matrix[season_id], self.season_matrix[season_id])

        diff_matrix = np.abs(pred[:, :, season_id] - self.train_tensor[:, :, season_id])
        m = (diff_matrix * self.mask_matrix)
        app_error = np.zeros(self.num_appliance)
        for i in range(1, self.num_appliance):
            tmp = m[:, i]
            if len(tmp[tmp.nonzero()]) != 0:
                app_error[i] = tmp[tmp.nonzero()].mean()
        return app_error

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
