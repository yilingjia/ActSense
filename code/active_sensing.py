from algo import *
from basic import *
import pickle
import os
import argparse


def active_sensing(year=2015, dataset='filter', method='active', latent_dimension=3, alpha1=0.1, alpha2=0.1,
                   alpha3=0.1, lambda1=10000, lambda2=10000, lambda3=10000, gamma1=0.5, gamma2=0.5, init='random',
                   uncertainty='current', kernel='gaussian', sigma=12, k=5, normalization='none', random_seed=0,
                   iters=5, season_type='fixed', reg=True):

    AS = {}
    app_matrix = {}
    season_matrix = {}
    selected_pair = {}
    test_home_matrix = {}
    train_home_matrix = {}
    uncty = {}
    # learn the season factor from the aggregate readings of last year
    pre_tensor, homeids = get_tensor(year - 1, dataset)
    pre_tensor[:, 1:] = np.NaN
    T = np.ones(12).reshape(-1, 1)
    h, app, season = factorization(pre_tensor, num_latent=latent_dimension, dis=True, random_seed=random_seed, T_known=T)

    tensor, homeids = get_tensor(year, dataset)
    # for each fold, do the active learning
    for fold_num in range(5):

        print("Fold: ", fold_num)
        train, test, tr_homes, tt_homes = get_train_test(tensor, homeids, fold_num=fold_num)
        if init == 'random':
            season = None
        AS[fold_num] = ActiveSensing_fix(train_tensor=train, test_tensor=test, init_list=None, reg=reg, pre_app=None,
                                         pre_season=season, method=method, latent_dimension=latent_dimension,
                                         alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, lambda1=lambda1, lambda2=lambda2,
                                         lambda3=lambda3, gamma1=gamma1,
                                         gamma2=gamma2, init=init,
                                         uncertainty=uncertainty, kernel=kernel, sigma=sigma, k=k, normalization=normalization, random_seed=random_seed)
        for t in range(12):
            AS[fold_num].select(t)
            # learn home and appliance factor from random values
            if season_type == 'fixed':
                AS[fold_num].update_als_fix_season(t, iters)
            else:
                AS[fold_num].update_als_update_season(t, iters)
            print("In iteration {}, Observed items: ".format(t), AS[fold_num].ob_list_length[t])
            # AS[fold_num].display_train_error(t)
        app_matrix[fold_num] = AS[fold_num].app_matrix
        season_matrix[fold_num] = AS[fold_num].season_matrix
        selected_pair[fold_num] = AS[fold_num].selected_pair
        test_home_matrix[fold_num] = AS[fold_num].test_home_matrix
        train_home_matrix[fold_num] = AS[fold_num].train_home_matrix
        uncty[fold_num] = AS[fold_num].uncertainty
        # print(AS[fold_num].uncertainty[0][0][0])

    # get the prediction
    m, n, o = tensor.shape
    pred = np.empty((12, m, n, o))
    for t in range(12):
        # pred[t] = np.empty(tensor.shape)
        num_home = 0
        for fold_num in range(5):
            pr = np.einsum('Hr, Ar, Sr -> HAS', AS[fold_num].test_home_matrix[t],
                           AS[fold_num].app_matrix[t], AS[fold_num].season_matrix[t])
            pred[t][num_home:(num_home + AS[fold_num].test_home_matrix[t].shape[0])] = pr
            num_home += AS[fold_num].test_home_matrix[t].shape[0]

    return pred, train_home_matrix, test_home_matrix, app_matrix, season_matrix, selected_pair, uncty


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", help="the year of the energy data", default=2015)
    parser.add_argument("--dataset", help="the type of dataset", default='artificial')
    parser.add_argument("--method", help="selection method", default='active')
    parser.add_argument("--init", help="initialization method (pre, random)", default="pre")
    parser.add_argument("--uncertainty", help="uncertainty method(one, next, equal, weighted)", default="prev_future_weighted")
    parser.add_argument("--alpha1", help="alpha1 value", default=0.1)
    parser.add_argument("--alpha2", help="alpha2 value", default=0.1)
    parser.add_argument("--alpha3", help="alpha3 value", default=0)
    parser.add_argument("--k", help="number of selection at each trial", default=5)
    parser.add_argument("--latent_dimension", help="number of latent dimension", default=4)
    parser.add_argument("--season_type", help="updated or fixed season factor", default="updated")
    parser.add_argument("--regularization", help="true (new version) or false (previous version)", default="true")
    parser.add_argument("--gamma1", help="weight of current uncertainty", default=0.5)
    parser.add_argument("--gamma2", help="weight of uncertianty", default=0.5)
    parser.add_argument("--lambda1", help="lambda1 for home factor", default=10000)
    parser.add_argument("--lambda2", help="lambda2 for appliance factor", default=10000)
    parser.add_argument("--lambda3", help="lambda3 for season factor", default=10000)
    parser.add_argument("--kernel", help="kernel function for the weight", default="gaussian")
    parser.add_argument("--sigma", help="sigma value for kernel function", default=12)

    args = parser.parse_args()
    print(args)

    # initialize the parameters
    iters = 10
    lambda1 = int(args.lambda1)
    lambda2 = int(args.lambda2)
    lambda3 = int(args.lambda3)
    year = int(args.year)
    dataset = args.dataset
    method = args.method
    init = args.init
    uncertainty = args.uncertainty
    alpha1 = float(args.alpha1)
    alpha2 = float(args.alpha2)
    alpha3 = float(args.alpha3)
    latent_dimension = int(args.latent_dimension)
    k = int(args.k)
    season_type = args.season_type
    kernel = args.kernel
    sigma = int(args.sigma)
    reg = (args.regularization == "true")
    gamma1 = float(args.gamma1)
    gamma2 = float(args.gamma2)
    num_random = 10
    normalization = "none"

    # prepare for the algorithm
    tensor, hid = get_tensor(year, dataset)
    m, n, o = tensor.shape
    prediction = np.empty((num_random, 12, m, n, o))
    appliance_m = {}
    season_m = {}
    train_h_m = {}
    test_h_m = {}
    sp = {}
    uncty = {}

    for random_seed in range(num_random):
        print("random seed: ", random_seed)
        print("uncertainty: ", uncertainty)
        pred, train_home_matrix, test_home_matrix, app_matrix, season_matrix, selected_pair, uty = active_sensing(year=year, dataset=dataset,
                                                                                                                  method=method,
                                                                                                                  latent_dimension=latent_dimension,
                                                                                                                  alpha1=alpha1,
                                                                                                                  alpha2=alpha2,
                                                                                                                  alpha3=alpha3,
                                                                                                                  lambda1=lambda1,
                                                                                                                  lambda2=lambda2,
                                                                                                                  lambda3=lambda3,
                                                                                                                  init=init,
                                                                                                                  uncertainty=uncertainty,
                                                                                                                  kernel=kernel,
                                                                                                                  sigma=sigma,
                                                                                                                  k=k,
                                                                                                                  normalization=normalization,
                                                                                                                  random_seed=random_seed,
                                                                                                                  iters=iters,
                                                                                                                  season_type=season_type,
                                                                                                                  reg=reg)
        prediction[random_seed] = pred
        season_m[random_seed] = season_matrix
        appliance_m[random_seed] = app_matrix
        sp[random_seed] = selected_pair
        train_h_m[random_seed] = train_home_matrix
        test_h_m[random_seed] = test_home_matrix
        uncty[random_seed] = uty

    if reg:
        pre_dir = "../data/result/reg"
    else:
        pre_dir = "../data/result"

    if season_type == 'fixed':
        directory = "{}/fixed_season/{}/{}/{}/".format(pre_dir, method, year, dataset)
    else:
        directory = "{}/update_season/{}/{}/{}/".format(pre_dir, method, year, dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)

    param_name = "{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(init, normalization, uncertainty,
                                                                 kernel, sigma, alpha1, alpha2, alpha3,
                                                                 lambda1, lambda2, lambda3, k, latent_dimension)

    e_file = "pred-{}".format(param_name)
    sp_file = "sp-{}".format(param_name)
    am_file = "app-matrix-{}".format(param_name)
    sm_file = "season-matrix-{}".format(param_name)
    tr_file = "train-matrix-{}".format(param_name)
    tt_file = "test-matrix-{}".format(param_name)
    ut_file = "uncertainty-{}".format(param_name)
    print(directory + e_file)
    f = open(directory + e_file, 'wb')
    pickle.dump(prediction, f)
    f.close()

    print(directory + am_file)
    f = open(directory + am_file, "wb")
    pickle.dump(appliance_m, f)
    f.close()

    print(directory + sm_file)
    f = open(directory + sm_file, "wb")
    pickle.dump(season_m, f)
    f.close()

    print(directory + sp_file)
    f = open(directory + sp_file, "wb")
    pickle.dump(sp, f)
    f.close()

    print(directory + tr_file)
    f = open(directory + tr_file, "wb")
    pickle.dump(train_h_m, f)
    f.close()

    print(directory + tt_file)
    f = open(directory + tt_file, "wb")
    pickle.dump(test_h_m, f)
    f.close()

    print(directory + ut_file)
    f = open(directory + ut_file, "wb")
    pickle.dump(uncty, f)
    f.close()
