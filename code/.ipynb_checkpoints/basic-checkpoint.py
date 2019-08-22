import autograd.numpy as np
from autograd import grad
from autograd import multigrad
from autograd.numpy import linalg as LA
from sklearn.model_selection import KFold

APPLIANCE_ORDER_OBSERVED = ['use', 'air1', 'clotheswasher1', 'dishwasher1', 'furnace1', 'microwave1', 'refrigerator1']
APPLIANCE_ORDER_MISSING = ['use', 'air1', 'clotheswasher1', 'drye1', 'dishwasher1', 'furnace1', 'kitchenapp1','microwave1','refrigerator1']

def get_tensor(year=2015, data_type='missing'):
	if data_type == 'missing':
		raw_data = np.load("../data/data-2013-2017-missing.npy").item()
		ORDER = APPLIANCE_ORDER_MISSING
	if data_type == 'filter':
		raw_data = np.load("../data/data-2013-2017-missing-filtered.npy").item()
		ORDER = APPLIANCE_ORDER_MISSING
	if data_type == 'observed':
		raw_data = np.load("../data/data-2013-2017-observed-filtered.npy").item()
		ORDER = APPLIANCE_ORDER_OBSERVED

	data_year = raw_data[year]
	tensor = np.empty((len(data_year.keys()), len(ORDER), 12))
	tensor[:] = np.NaN
	for i, home_id in enumerate(data_year.keys()):
		for j, appliance in enumerate(ORDER):
			try:
				tensor[i, j, :12] = data_year[home_id][appliance].values
			except:
				continue

	return tensor, list(data_year.keys())

def get_sub_tensor(year, home_ids, data_type):
    if data_type == 'missing':
        raw_data = np.load("../data/data-2013-2017-missing.npy").item()
        ORDER = APPLIANCE_ORDER_MISSING
    if data_type == 'filter':
        raw_data = np.load("../data/data-2013-2017-missing-filtered.npy").item()
        ORDER = APPLIANCE_ORDER_MISSING
    if data_type == 'observed':
        raw_data = np.load("../data/data-2013-2017-observed-filtered.npy").item()
        ORDER = APPLIANCE_ORDER_OBSERVED
        
    data_year = raw_data[year]
    tensor = np.empty((len(home_ids), len(ORDER), 12))
    tensor[:] = np.NaN
    for i, home_id in enumerate(home_ids):
        for j, appliance in enumerate(ORDER):
            try:
                tensor[i, j, :12] = data_year[home_id][appliance].values
            except:
                continue

    return tensor

def cost_abs(H, A, S, E_np_masked):
    HAT = np.einsum('Hr, Ar, Sr -> HAS', H, A, S)
    mask = ~np.isnan(E_np_masked)
    error_1 = (HAT - E_np_masked)[mask].flatten() 
    error_2 = 0.01*LA.norm(H) + 0.01*LA.norm(A) + 0.01*LA.norm(S)
    return np.sqrt((error_1 ** 2).mean()) + error_2

def set_known(A, W):
    mask = ~np.isnan(W)
    A[:, :mask.shape[1]][mask] = W[mask]
    return A

def factorization(tensor, num_latent, num_iter=2000, lr=1, dis=False, random_seed=0, eps=1e-8, T_known=None):
    np.random.seed(random_seed)
    cost = cost_abs
    
    args_num = [0, 1, 2]
    mg = multigrad(cost, argnums = args_num)
    M, N, O = tensor.shape

    H = np.random.rand(M, num_latent)
    A = np.random.rand(N, num_latent)
    T = np.random.rand(O, num_latent)

    sum_square_gradients_A = np.zeros_like(A)
    sum_square_gradients_H = np.zeros_like(H)
    sum_square_gradients_T = np.zeros_like(T)
    if T_known is not None:
        T = set_known(T, T_known)
    
    # GD procedure
    for i in range(num_iter):
        del_h, del_a, del_t = mg(H, A, T, tensor)
        
        sum_square_gradients_A += eps + np.square(del_a)
        lr_a = np.divide(lr, np.sqrt(sum_square_gradients_A))
        A -= lr_a * del_a
        
        sum_square_gradients_H += eps + np.square(del_h)
        sum_square_gradients_T += eps + np.square(del_t)

        lr_h = np.divide(lr, np.sqrt(sum_square_gradients_H))
        lr_t = np.divide(lr, np.sqrt(sum_square_gradients_T))

        H -= lr_h * del_h
        T -= lr_t * del_t
        
        if T_known is not None:
            T = set_known(T, T_known)


        # Projection to non-negative space
        H[H < 0] = 1e-8
        A[A < 0] = 1e-8
        T[T < 0] = 1e-8

        if i % 500 == 0:
            if dis:
                print(cost(H, A, T, tensor))
    
    return H, A, T

# create list for the tensor observations
def obs_list(tensor):
	obs = {}
	mask_tensor = ~np.isnan(tensor)
	M, N, O = tensor.shape
	c = 0
	for i in range(M):
		for j in range(N):
			for k in range(O):
				if mask_tensor[i, j, k]:
					obs[c] = [i, j, k, tensor[i, j, k]]
					c += 1
	return obs

def ALS(tensor, num_latent=3, random_seed=0, lambda1=10000, lambda2=10000, lambda3=10000, iters=10):
	np.random.seed(random_seed)
	mask_tensor = ~np.isnan(tensor)

	# initialization
	num_homes, num_appliance, num_season = tensor.shape
	A = np.empty([num_homes, num_latent, num_latent])
	b = np.empty([num_homes, num_latent, 1])
	h = np.empty([num_homes, num_latent, 1])
	for i in range(num_homes):
		A[i] = np.identity(num_latent)*lambda1
		b[i] = np.zeros(num_latent).reshape(-1, 1)
		h[i] = np.random.uniform(0, 50, num_latent).reshape(-1, 1)
		h[i] = np.ones(num_latent).reshape(-1, 1)

	C = np.empty([num_appliance, num_latent, num_latent])
	d = np.empty([num_appliance, num_latent, 1])
	a = np.empty([num_appliance, num_latent, 1])
	for i in range(num_appliance):
		C[i] = np.identity(num_latent)*lambda2
		d[i] = np.zeros(num_latent).reshape(-1, 1)
		a[i] = np.random.uniform(0, 50, num_latent).reshape(-1, 1)
		a[i] = np.ones(num_latent).reshape(-1, 1)
		
	E = np.empty([num_season, num_latent, num_latent])
	f = np.empty([num_season, num_latent, 1])
	s = np.empty([num_season, num_latent, 1])
	for t in range(num_season):
		E[t] = np.identity(num_latent)*lambda3
		f[t] = np.zeros(num_latent).reshape(-1, 1)
		s[t] = np.random.uniform(0, 50, num_latent).reshape(-1, 1)
		s[t] = np.ones(num_latent).reshape(-1, 1)

	# get the observed list
	observed_list = obs_list(tensor)
	
	for iters in range(20):
		np.random.shuffle(observed_list)
		for i in observed_list:
			hid, aid, sid, e = observed_list[i]
			E[sid] += np.matmul(np.multiply(a[aid], h[hid]), np.multiply(a[aid], h[hid]).transpose())
			f[sid] += e * (np.multiply(a[aid], h[hid]))
			s[sid] = np.matmul(np.linalg.inv(E[sid]), f[sid])
			s[s < 0] = 1e-8

		for i in observed_list:
			hid, aid, sid, e = observed_list[i]
			A[hid] += np.matmul(np.multiply(a[aid], s[sid]), np.multiply(a[aid], s[sid]).transpose())
			b[hid] += e * (np.multiply(a[aid], s[sid]))
			h[hid] = np.matmul(np.linalg.inv(A[hid]), b[hid])
			h[h < 0] = 1e-8

		for i in observed_list:
			hid, aid, sid, e = observed_list[i]
			C[aid] += np.matmul(np.multiply(h[hid], s[sid]), np.multiply(h[hid], s[sid]).transpose())
			d[aid] += e * (np.multiply(h[hid], s[sid]))
			a[aid] = np.matmul(np.linalg.inv(C[aid]), d[aid])
			a[a < 0] = 1e-8

	home = {}
	app = {}
	season = {}
	for i in range(num_homes):
		home[i] = {}
		home[i]['A'] = A[i]
		home[i]['b'] = b[i]
		home[i]['h'] = h[i]
	for i in range(num_appliance):
		app[i] = {}
		app[i]['C'] = C[i]
		app[i]['d'] = d[i]
		app[i]['a'] = a[i]
	for i in range(num_season):
		season[i] = {}
		season[i]['E'] = E[i]
		season[i]['f'] = f[i]
		season[i]['s'] = s[i]
			
	return home, app, season, h, a, s

def get_train_test(tensor, homeids, num_folds=5, fold_num=0):
	homeids = np.array(homeids)
	num_homes= tensor.shape[0]
	k = KFold(n_splits=num_folds)
	train, test = list(k.split(range(0, num_homes)))[fold_num]
	tr_ids, tt_ids = list(k.split(range(0, num_homes)))[fold_num]
	return tensor[train], tensor[test], list(homeids[tr_ids]), list(homeids[tt_ids])

def get_instrumented_data(current_year, data_type, tr_homes, k, random_seed):
    np.random.seed(random_seed)
    past_year = current_year - 1
    tensor, homeids = get_tensor(past_year, data_type)
    
    common_homes = set(homeids) & set(tr_homes)
    instrumented = np.random.choice(list(common_homes), k)
    
    # get instrumented tensor of past year
    inst_tensor = get_sub_tensor(past_year, instrumented, data_type)
    
    # get the index of instrumented homes of current year
    inst_idx = []
    for idx, hid in enumerate(instrumented):
        inst_idx.append(list(tr_homes).index(hid))
    
    return inst_tensor, inst_idx