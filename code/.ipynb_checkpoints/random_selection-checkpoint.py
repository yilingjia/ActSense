from algo_fix_season import *
from basic import *
# import pandas as pd
import pickle
import os


def active_sensing(year=2015, dataset='filter', method='active', latent_dimension=3, alpha1=0.1, alpha2=0.1, alpha3=0.1,
                   init='random', uncertainty='one', k=5, normalization='none', random_seed=0, iters=5, season_type='fixed', reg=False):

    AS = {}
    app_matrix = {}
    season_matrix = {}
    selected_pair = {}
    train_home_matrix = {}
    test_home_matrix = {}

    # learn the season factor from the aggregate readings of last year
    pre_tensor, homeids = get_tensor(year - 1, dataset)
    pre_tensor[:, 1:] = np.NaN
    T = np.ones(12).reshape(-1, 1)
    h, app, season = factorization(pre_tensor, num_latent=latent_dimension, dis=True, random_seed=random_seed, T_known=T)

    tensor, homeids = get_tensor(year, dataset)
    # for each fold, do the active learning
    for fold_num in range(5):

        # print("Fold: ", fold_num)
        train, test, tr_homes, tt_homes = get_train_test(tensor, homeids, fold_num=fold_num)
        if init == 'random':
            season = None
        AS[fold_num] = ActiveSensing_fix(train_tensor=train, test_tensor=test, init_list=None, pre_app=None, pre_season=season, reg=reg,
                                         method=method, latent_dimension=latent_dimension, alpha1=alpha1, alpha2=alpha2, alpha3=alpha3,
                                         init=init, uncertainty=uncertainty, k=k,
                                         normalization=normalization, random_seed=random_seed)
        for t in range(12):
            AS[fold_num].select(t)
            # learn home and appliance factor from random values
            if season_type == 'fixed':
                AS[fold_num].update_als_fix_season(t, iters)
            else:
                AS[fold_num].update_als_update_season(t, iters)
            # print("In iteration {}, Observed items: ".format(t), AS[fold_num].ob_list_length[t])
            # AS[fold_num].display_train_error(t)
        app_matrix[fold_num] = AS[fold_num].app_matrix
        season_matrix[fold_num] = AS[fold_num].season_matrix
        selected_pair[fold_num] = AS[fold_num].selected_pair
        train_home_matrix[fold_num] = AS[fold_num].train_home_matrix
        test_home_matrix[fold_num] = AS[fold_num].test_home_matrix

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

    return pred, train_home_matrix, test_home_matrix, app_matrix, season_matrix, selected_pair


if __name__ == "__main__":

    year, dataset, init, k, latent_dimension, season_type, reg = sys.argv[1:]

    num_random = 10
    year = int(year)
    latent_dimension = int(latent_dimension)
    k = int(k)
    reg = (reg == 'true')
    print(reg)

    tensor, hid = get_tensor(year, dataset)
    m, n, o = tensor.shape

    prediction = np.empty((num_random, 12, m, n, o))
    appliance_m = {}
    season_m = {}
    train_h_m = {}
    test_h_m = {}
    sp = {}

    for random_seed in range(num_random):
        print("random seed: ", random_seed)
        pred, train_home_matrix, test_home_matrix, app_matrix, season_matrix, selected_pair = active_sensing(year=year,
                                                                                                             dataset=dataset,
                                                                                                             method='random',
                                                                                                             latent_dimension=latent_dimension,
                                                                                                             alpha1=0.1,
                                                                                                             alpha2=0.1,
                                                                                                             alpha3=0.1,
                                                                                                             init=init,
                                                                                                             lambda_=10000,
                                                                                                             uncertainty='one',
                                                                                                             k=k,
                                                                                                             normalization='none',
                                                                                                             random_seed=random_seed,
                                                                                                             iters=5,
                                                                                                             season_type=season_type,
                                                                                                             reg=reg)

        prediction[random_seed] = pred
        season_m[random_seed] = season_matrix
        appliance_m[random_seed] = app_matrix
        sp[random_seed] = selected_pair
        train_h_m[random_seed] = train_home_matrix
        test_h_m[random_seed] = test_home_matrix

    if reg:
        pre_dir = "../data/reg"
    else:
        pre_dir = "../data"

    if season_type == 'fixed':
        directory = "{}/fixed_season/random/{}/{}/".format(pre_dir, year, dataset)
    else:
        directory = "{}/update_season/random/{}/{}/".format(pre_dir, year, dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)

    pred_file = 'pred-{}-{}-{}'.format(init, k, latent_dimension)
    sp_file = 'sp-{}-{}-{}'.format(init, k, latent_dimension)
    am_file = 'app-matrix-{}-{}-{}'.format(init, k, latent_dimension)
    sm_file = 'season-matrix-{}-{}-{}'.format(init, k, latent_dimension)
    tr_file = 'train-matrix-{}-{}-{}'.format(init, k, latent_dimension)
    tt_file = 'test-matrix-{}-{}-{}'.format(init, k, latent_dimension)

    f = open(directory + pred_file, 'wb')
    pickle.dump(prediction, f)
    f.close()
    f = open(directory + am_file, "wb")
    pickle.dump(appliance_m, f)
    f.close()
    f = open(directory + sm_file, "wb")
    pickle.dump(season_m, f)
    f.close()
    f = open(directory + sp_file, "wb")
    pickle.dump(sp, f)
    f.close()
    f = open(directory + tr_file, "wb")
    pickle.dump(train_h_m, f)
    f.close()
    f = open(directory + tt_file, "wb")
    pickle.dump(test_h_m, f)
    f.close()
