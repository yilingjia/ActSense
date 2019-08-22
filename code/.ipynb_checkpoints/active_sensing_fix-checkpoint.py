from algo_fix_season import *
from basic import *
# import pandas as pd
import pickle
import os


def active_sensing(year=2015, dataset='filter', method='active', latent_dimension=3, alpha1=0.1, alpha2=0.1, alpha3=0.1, init='random', uncertainty='one', k=5, normalization='none', random_seed=0, iters=5, season_type='fixed', reg=False):

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
        AS[fold_num] = ActiveSensing_fix(train_tensor=train, test_tensor=test, init_list=None, reg=reg, pre_app=None, pre_season=season,
                                         method=method, latent_dimension=latent_dimension, alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, init=init, uncertainty=uncertainty, k=k,
                                         normalization=normalization, random_seed=random_seed)
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

    year, dataset, method, init, normalization, uncertainty, alpha1, alpha2, alpha3, k, latent_dimension, season_type, reg = sys.argv[1:]

    iters = 5
    year = int(year)
    alpha1 = float(alpha1)
    alpha2 = float(alpha2)
    alpha3 = float(alpha3)
    latent_dimension = int(latent_dimension)
    k = int(k)
    reg = (reg == "true")

    num_random = 10
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
        # print(uty[0][0][0])

    # print("OK here")
    if reg:
        pre_dir = "../data/reg"
    else:
        pre_dir = "../data"

    if season_type == 'fixed':
        directory = "{}/fixed_season/{}/{}/{}/".format(pre_dir, method, year, dataset)
    else:
        directory = "{}/update_season/{}/{}/{}/".format(pre_dir, method, year, dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)

    e_file = "pred-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(init, normalization, uncertainty, lambda1, lambda2, lambda3, alpha1, alpha2, alpha3, k, latent_dimension)
    sp_file = "sp-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(init, normalization, uncertainty, lambda1, lambda2, lambda3, alpha1, alpha2, alpha3, k, latent_dimension)
    am_file = "app-matrix-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(init, normalization, uncertainty, lambda1, lambda2, lambda3, alpha1, alpha2, alpha3, k, latent_dimension)
    sm_file = "season-matrix-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(init, normalization, uncertainty, lambda1, lambda2, lambda3, alpha1, alpha2, alpha3, k, latent_dimension)
    tr_file = "train-matrix-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(init, normalization, uncertainty, lambda1, lambda2, lambda3, alpha1, alpha2, alpha3, k, latent_dimension)
    tt_file = "test-matrix-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(init, normalization, uncertainty, lambda1, lambda2, lambda3, alpha1, alpha2, alpha3, k, latent_dimension)
    ut_file = "uncertainty-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(init, normalization, uncertainty, lambda1, lambda2, lambda3, alpha1, alpha2, alpha3, k, latent_dimension)
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
