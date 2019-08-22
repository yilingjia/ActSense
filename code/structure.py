import numpy as np


class ActiveSensingHomeStruct:
    def __init__(self, id, latent_dimension, lambda_, init="zero", reg=False, random_seed=0):
        np.random.seed(random_seed)
        self.id = id
        self.latent_dimension = latent_dimension
        self.A = lambda_ * np.identity(n=self.latent_dimension)
        if reg:
            self.b = np.zeros(self.latent_dimension).reshape(-1, 1) + lambda_
        else:
            self.b = np.zeros(self.latent_dimension).reshape(-1, 1)
        self.AInv = np.linalg.inv(self.A)
        self.time = 0
        self.fakeA = self.A.copy()
        self.fakeAInv = np.linalg.inv(self.fakeA)

        if init == 'zero':
            self.h = np.zeros(self.latent_dimension).reshape(-1, 1)
        else:
            self.h = np.random.rand(self.latent_dimension, 1)

    def update_parameters(self, app, season, energy):
        self.time += 1
        self.A += np.matmul(np.multiply(app.a, season.s), np.multiply(app.a, season.s).transpose())
        self.b += energy * (np.multiply(app.a, season.s))
        self.AInv = np.linalg.inv(self.A)
        self.h = np.dot(self.AInv, self.b)
        self.h[self.h < 0] = 1e-8

    def fake_update_parameters(self, app, season):
        self.fakeA = self.A + np.matmul(np.multiply(app.a, season.s), np.multiply(app.a, season.s).transpose())
        self.fakeAInv = np.linalg.inv(self.fakeA)


class ActiveSensingAppStruct:
    def __init__(self, id, latent_dimension, lambda_, init="zero", reg=False, pre_app=None, random_seed=0):
        np.random.seed(random_seed)
        self.id = id
        self.latent_dimension = latent_dimension
        self.C = lambda_ * np.identity(n=self.latent_dimension)
        if reg:
            self.d = np.zeros(self.latent_dimension).reshape(-1, 1) + lambda_
        else:
            self.d = np.zeros(self.latent_dimension).reshape(-1, 1)
        self.CInv = np.linalg.inv(self.C)
        self.time = 0
        self.fakeC = self.C.copy()
        self.fakeCInv = np.linalg.inv(self.fakeC)

        if init == 'random' or (init == 'pre' and pre_app is None):
            self.a = np.random.rand(self.latent_dimension, 1)
        if init == 'zero':
            self.a = np.zeros(self.latent_dimension).reshape(-1, 1)
        if init == 'pre' and pre_app is not None:
            self.a = pre_app.reshape(-1, 1)

    def update_parameters(self, home, season, energy):
        self.time += 1
        self.C += np.matmul(np.multiply(home.h, season.s), np.multiply(home.h, season.s).transpose())
        self.d += energy * (np.multiply(home.h, season.s))
        self.CInv = np.linalg.inv(self.C)
        self.a = np.dot(self.CInv, self.d)
        self.a[self.a < 0] = 1e-8

    def fake_update_parameters(self, home, season):
        self.fakeC = self.C + np.matmul(np.multiply(home.h, season.s), np.multiply(home.h, season.s).transpose())
        self.fakeCInv = np.linalg.inv(self.fakeC)


class ActiveSensingSeasonStruct:
    def __init__(self, id, latent_dimension, lambda_, init="zero", reg=False, pre_season=None, random_seed=0):
        np.random.seed(random_seed)
        self.id = id
        self.latent_dimension = latent_dimension
        self.E = lambda_ * np.identity(n=self.latent_dimension)
        if reg:
            self.f = np.zeros(self.latent_dimension).reshape(-1, 1) + lambda_ * pre_season.reshape(-1, 1)
        else:
            self.f = np.zeros(self.latent_dimension).reshape(-1, 1) + lambda_ * pre_season.reshape(-1, 1)
        self.EInv = np.linalg.inv(self.E)
        self.time = 0
        self.fakeE = self.E.copy()
        self.fakeEInv = np.linalg.inv(self.fakeE)

        if init == 'random' or (init == 'pre' and pre_season is None):
            self.s = np.random.rand(self.latent_dimension, 1)
        if init == 'zero':
            self.s = np.zeros(self.latent_dimension).reshape(-1, 1)
        if init == 'pre' and pre_season is not None:
            self.s = pre_season.reshape(-1, 1)

    def update_parameters(self, home, app, energy):
        self.time += 1
        self.E += np.matmul(np.multiply(home.h, app.a), np.multiply(home.h, app.a).transpose())
        self.f += energy * (np.multiply(home.h, app.a))
        self.EInv = np.linalg.inv(self.E)
        self.s = np.dot(self.EInv, self.f)
        self.s[self.s < 0] = 1e-8

    def update_parameters_2(self, home, app, energy):
        self.time += 1
        self.E += np.matmul(np.multiply(home.h, app.a), np.multiply(home.h, app.a).transpose())
        self.EInv = np.linalg.inv(self.E)

    def fake_update_parameters(self, home, app):
        self.fakeE = self.E + np.matmul(np.multiply(home.h, app.a), np.multiply(home.h, app.a).transpose())
        self.fakeEInv = np.linalg.inv(self.fakeE)
