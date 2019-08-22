from structure import *
# from basic import get_appliance_order
from sklearn.metrics import mean_squared_error
# from scipy import stats
# import sys
# import pandas as pd


class QueryByCommittee:
    def __init__(self, train_tensor, test_tensor, pre_season=None, lambda1=10000, lambda2=10000, lambda3=10000,
                 init='random', reg=False, latent_dimension=3, k=5, random_seed=0):

        np.random.seed(random_seed)
        self.train_tensor = train_tensor
        self.test_tensor = test_tensor
        self.init = init
        self.latent_dimension = latent_dimension
        self.k = k
        self.random_seed = random_seed
        self.num_train, self.num_appliance, self.num_season = train_tensor.shape
        self.num_test = test_tensor.shape[0]
        self.mask_matrix = np.ones((self.num_train, self.num_appliance), dtype=int)
        self.pred_tensor = {}
        self.num_member = 6
        self.ob_list = {}
        self.ob_list_length = {}
        self.selected_pair = {}
        self.std = {}

        self.train_home_matrix = {}
        self.test_home_matrix = {}
        self.app_matrix = {}
        self.season_matrix = {}
        for t in range(12):
            self.train_home_matrix[t] = np.zeros((self.num_train, self.latent_dimension))
            self.test_home_matrix[t] = np.zeros((self.num_test, self.latent_dimension))
            self.app_matrix[t] = np.zeros((self.num_appliance, self.latent_dimension))
            self.season_matrix[t] = np.zeros((self.num_season, self.latent_dimension))
        self.train_errors = {}
        self.test_errors = {}
        self.train_errors_per_month = {}
        self.test_errors_per_month = {}
        self.selected_pair = {}
        for i in range(self.num_season):
            self.train_errors_per_month[i] = {}
            self.test_errors_per_month[i] = {}

        for t in range(12):
            self.ob_list_length[t] = 0

        # create committee members
        self.cm = {}
        for i, latent_dimension in enumerate(range(2, 8)):
            if pre_season != None:
                self.cm[latent_dimension] = CommitteeMember(train_tensor=train_tensor, test_tensor=test_tensor,
                                                            pre_season=pre_season[latent_dimension],
                                                            latent_dimension=latent_dimension, lambda1=lambda1,
                                                            lambda2=lambda2, lambda3=lambda3,
                                                            init=init, reg=reg, k=k, random_seed=random_seed)
            else:
                self.cm[latent_dimension] = CommitteeMember(train_tensor=train_tensor, test_tensor=test_tensor,
                                                            pre_season=None, latent_dimension=latent_dimension,
                                                            lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
                                                            init=init, reg=reg, k=k, random_seed=random_seed)

        for i in range(12):
            self.pred_tensor[i] = np.zeros((self.num_member, self.num_train, self.num_appliance))

    def get_prediction_tensor(self, season_id):
        for i, latent_dimension in enumerate(range(2, 8)):
            self.pred_tensor[season_id][i] = np.einsum('Hr, Ar, Sr -> HAS',
                                                       self.cm[latent_dimension].train_home_matrix[season_id],
                                                       self.cm[latent_dimension].app_matrix[season_id],
                                                       self.cm[latent_dimension].season_matrix[season_id])[:, :, season_id]
            # print(self.pred_tensor[season_id][i])

    def select(self, season_id):
        # get the prediction tensor of current month
        if season_id == 0:
            self.selected_pair[season_id] = {}
            self.mask_matrix[:, 0] = 0
            mask_array = self.mask_matrix.flatten()
            np.random.seed(self.random_seed)
            random_candidate = []
            for i, val in enumerate(mask_array):
                if val == 1:
                    random_candidate.append(i)
            index = np.random.choice(random_candidate, self.k)
            home_index, appliance_index = divmod(index, self.num_appliance)

            for i in range(len(home_index)):
                self.mask_matrix[home_index[i], appliance_index[i]] = 0
                self.selected_pair[season_id][i] = [home_index[i], appliance_index[i]]

        else:
            self.selected_pair[season_id] = {}
            self.get_prediction_tensor(season_id - 1)
            self.std[season_id - 1] = np.std(self.pred_tensor[season_id - 1], axis=0)
            std_matrix = np.multiply(self.std[season_id - 1], self.mask_matrix)
            home_index, appliance_index = np.unravel_index(np.argsort(std_matrix.ravel())[-self.k:],
                                                           std_matrix.shape)

            for i in range(len(home_index)):
                self.mask_matrix[home_index[i], appliance_index[i]] = 0
                self.selected_pair[season_id][i] = [home_index[i], appliance_index[i]]

        self.update_observed_list(season_id)

    def update_observed_list(self, season_id):
        # update the observed list with the training set
        if season_id != 0:
            self.ob_list_length[season_id] = self.ob_list_length[season_id - 1]
        for i in range(self.num_train):
            for j in range(self.num_appliance):
                if self.mask_matrix[i, j] == 0 and ~np.isnan(self.train_tensor[i, j, season_id]):
                    self.ob_list[self.ob_list_length[season_id]] = [i, j, season_id, self.train_tensor[i, j, season_id]]
                    self.ob_list_length[season_id] += 1

        # update the observed list with the test set: only aggregate
        for i in range(self.num_test):
            if ~np.isnan(self.test_tensor[i, 0, season_id]):
                self.ob_list[self.ob_list_length[season_id]] = [i + self.num_train, 0, season_id, self.test_tensor[i, 0, season_id]]
                self.ob_list_length[season_id] += 1

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
        # HA = np.einsum('Hr, Ar -> HAr', self.test_home_matrix, self.appd_matrix)
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

    def update_matrix(self, season_id):
        for i in range(self.num_train):
            self.train_home_matrix[season_id][i] = self.cm[self.latent_dimension].homes[i].h.reshape(1, self.latent_dimension)
        for i in range(self.num_test):
            self.test_home_matrix[season_id][i] = self.cm[self.latent_dimension].homes[i + self.num_train].h.reshape(1, self.latent_dimension)
        for i in range(self.num_appliance):
            self.app_matrix[season_id][i] = self.cm[self.latent_dimension].apps[i].a.reshape(1, self.latent_dimension)
        for i in range(self.num_season):
            self.season_matrix[season_id][i] = self.cm[self.latent_dimension].seasons[i].s.reshape(1, self.latent_dimension)

    def update_cm_fix_season(self, season_id):
        for i in range(2, 8):
            self.cm[i].update_als_fix_season(season_id, self.ob_list)
        self.update_matrix(season_id)
        self.update_train_error(season_id)
        self.update_test_error(season_id)

    def update_cm_update_season(self, season_id):
        for i in range(2, 8):
            self.cm[i].update_als_update_season(season_id, self.ob_list)
        self.update_matrix(season_id)
        self.update_train_error(season_id)
        self.update_test_error(season_id)


class CommitteeMember:
    def __init__(self, train_tensor, test_tensor, pre_season=None, latent_dimension=3, lambda1=10000,
                 lambda2=10000, lambda3=10000, init='random', reg=False, k=5, random_seed=0):

        # check the initialization
        if init == 'pre':
            if pre_season[0].shape[0] != latent_dimension:
                print(">>>>EXIT: Dimension of pre_season is not equal to latent_dimension!")
                sys.exit(0)

        np.random.seed(random_seed)
        self.random_seed = random_seed
        self.num_train, self.num_appliance, self.num_season = train_tensor.shape
        self.num_test = test_tensor.shape[0]
        self.num_home = self.num_train + self.num_test
        self.pre_season = pre_season
        self.latent_dimension = latent_dimension
        self.train_tensor = train_tensor
        self.test_tensor = test_tensor
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.init = init
        self.reg = reg
        # self.k = k
        # self.ob_list_length = {}
        # self.ob_list = {}
        self.train_home_matrix = {}
        self.test_home_matrix = {}
        self.app_matrix = {}
        self.season_matrix = {}
        for t in range(12):
            self.train_home_matrix[t] = np.zeros((self.num_train, self.latent_dimension))
            self.test_home_matrix[t] = np.zeros((self.num_test, self.latent_dimension))
            self.app_matrix[t] = np.zeros((self.num_appliance, self.latent_dimension))
            self.season_matrix[t] = np.zeros((self.num_season, self.latent_dimension))
        self.train_errors = {}
        self.test_errors = {}
        self.train_errors_per_month = {}
        self.test_errors_per_month = {}
        self.selected_pair = {}
        for i in range(self.num_season):
            self.train_errors_per_month[i] = {}
            self.test_errors_per_month[i] = {}

        # initialize the latent factors
        self.homes = []
        for i in range(self.num_home):
            self.homes.append(ActiveSensingHomeStruct(i, self.latent_dimension, lambda1, init, reg, random_seed=random_seed))

        self.apps = []
        for i in range(self.num_appliance):
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

    # def update_observed_list(self):

    def update_als_fix_season(self, season_id, ob_list, num_iterations=10):
        # Randomly init home factors
        self.homes = []
        for i in range(self.num_home):
            self.homes.append(ActiveSensingHomeStruct(i, self.latent_dimension, self.lambda1, self.init, self.reg,
                                                      random_seed=self.random_seed))
        # Randomly init appliance factors
        self.apps = []
        for i in range(self.num_appliance):
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

    def update_als_update_season(self, season_id, ob_list, num_iterations=10):

        # Randomly init home factors
        self.homes = []
        for i in range(self.num_home):
            self.homes.append(ActiveSensingHomeStruct(i, self.latent_dimension, self.lambda1, self.init, self.reg,
                                                      random_seed=self.random_seed))
        # Randomly init appliance factors
        self.apps = []
        for i in range(self.num_appliance):
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
            for i in ob_list:
                hid, aid, sid, e = ob_list[i]
                self.seasons[sid].update_parameters(self.homes[hid], self.apps[aid], e)
        # update corresponding matrices and the errors
        self.update_matrix(season_id)

    def update_matrix(self, season_id):
        for i in range(self.num_train):
            self.train_home_matrix[season_id][i] = self.homes[i].h.reshape(1, self.latent_dimension)
        for i in range(self.num_test):
            self.test_home_matrix[season_id][i] = self.homes[i + self.num_train].h.reshape(1, self.latent_dimension)
        for i in range(self.num_appliance):
            self.app_matrix[season_id][i] = self.apps[i].a.reshape(1, self.latent_dimension)
        for i in range(self.num_season):
            self.season_matrix[season_id][i] = self.seasons[i].s.reshape(1, self.latent_dimension)
