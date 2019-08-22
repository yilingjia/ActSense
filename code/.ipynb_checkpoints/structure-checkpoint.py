import numpy as np

class ActiveSensingHomeStruct:
	def __init__(self, id, latent_dimension, lambda_, init="zero", random_seed=0):
		np.random.seed(random_seed)
		self.id = id
		self.latent_dimension = latent_dimension

		self.A = lambda_*np.identity(n = self.latent_dimension)
		self.b = np.zeros(self.latent_dimension).reshape(-1 ,1)
		self.AInv = np.linalg.inv(self.A)

		self.time = 0

		if init == 'zero':
			# self.h = np.random.rand(self.latent_dimension)
			self.h = np.zeros(self.latent_dimension).reshape(-1 ,1)
			# self.h = np.random.uniform(0, 1, self.latent_dimension).reshape(-1, 1)
		else:
# 			self.h = np.random.uniform(0, 50, self.latent_dimension).reshape(-1, 1)
            self.h = np.random.rand(self.latent_dimension, 1)

	def updateParameters(self, app, season, energy):

		self.time += 1
		self.A += np.matmul(np.multiply(app.a, season.s), np.multiply(app.a, season.s).transpose())
		self.b += energy * (np.multiply(app.a, season.s))
		self.AInv = np.linalg.inv(self.A)

		self.h = np.dot(self.AInv, self.b)
		self.h[self.h < 0] = 1e-8

class ActiveSensingAppStruct:
	def __init__(self, id, latent_dimension, lambda_, init="zero", pre_app=None, random_seed=0):
		np.random.seed(random_seed)
		self.id = id
		self.latent_dimension = latent_dimension

		self.C = lambda_*np.identity(n = self.latent_dimension)
		self.d = np.zeros(self.latent_dimension).reshape(-1 ,1)
		self.CInv = np.linalg.inv(self.C)

		self.time = 0
		if init == 'random' or (init == 'pre' and pre_app is None):
			# self.a = np.random.rand(self.latent_dimension)
# 			self.a = np.random.uniform(0, 50, self.latent_dimension).reshape(-1 ,1)
            self.a = np.random.rand(self.latent_dimension, 1)
		if init == 'zero':
			self.a = np.zeros(self.latent_dimension).reshape(-1 ,1)
		if init == 'pre' and pre_app is not None:
			# self.a = pre_app['a'].reshape(-1, 1)
			# self.C = pre_app['C']
			# self.d = pre_app['d'].reshape(-1, 1)
			# self.CInv = np.linalg.inv(self.C)
			self.a = pre_app.reshape(-1, 1)

	def updateParameters(self, home, season, energy):

		self.time += 1
		self.C += np.matmul(np.multiply(home.h, season.s), np.multiply(home.h, season.s).transpose())
		self.d += energy * (np.multiply(home.h, season.s))
		self.CInv = np.linalg.inv(self.C)

		self.a = np.dot(self.CInv, self.d)
		self.a[self.a < 0] = 1e-8

class ActiveSensingSeasonStruct:
	def __init__(self, id, latent_dimension, lambda_, init="zero", pre_season=None, random_seed=0):
		np.random.seed(random_seed)
		self.id = id
		self.latent_dimension = latent_dimension

		self.E = lambda_*np.identity(n = self.latent_dimension)
		self.f = np.zeros(self.latent_dimension).reshape(-1 ,1)
		self.EInv = np.linalg.inv(self.E)

		self.time = 0
		if init == 'random' or (init == 'pre' and pre_season is None):
			# self.s = np.random.rand(self.latent_dimension)
# 			self.s = np.random.uniform(0, 50, self.latent_dimension).reshape(-1 ,1)
            self.s = np.random.rand(self.latent_dimension, 1)
		if init == 'zero':
			self.s = np.zeros(self.latent_dimension).reshape(-1 ,1)
		if init == 'pre' and pre_season is not None:
			# self.s = pre_season['s'].reshape(-1, 1)
			# self.E = pre_season['E']
			# self.f = pre_season['f'].reshape(-1, 1)
			# self.EInv = np.linalg.inv(self.E)
			self.s = pre_season.reshape(-1, 1)

	def updateParameters(self, home, app, energy):

		self.time += 1

		self.E += np.matmul(np.multiply(home.h, app.a), np.multiply(home.h, app.a).transpose())
		self.f += energy * (np.multiply(home.h, app.a))
		self.EInv = np.linalg.inv(self.E)

		self.s = np.dot(self.EInv, self.f)
		self.s[self.s < 0] = 1e-8
