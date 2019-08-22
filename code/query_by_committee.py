import sys
from algo_qbc import *
from basic import *
import numpy as np
import os
import pickle
import argparse


def query_by_committee(year=2015, dataset='filter', init='pre', reg=False, latent_dimension=3, lambda1=10000, lambda2=10000, lambda3=10000,
                       k=5, random_seed=0, iters=5, season_type='fixed'):

    QBC = {}
    app_matrix = {}
    season_matrix = {}
    selected_pair = {}
    test_home_matrix = {}
    train_home_matrix = {}

    pre_tensor, homeids = get_tensor(year - 1, dataset)
    pre_tensor[:, 1:] = np.NaN
    T = np.ones(12).reshape(-1, 1)
    season = {}
    for r in range(2, 8):
        h, app, season[r] = factorization(pre_tensor, num_latent=r, dis=False, random_seed=random_seed, T_known=T)

    # get the data of current year
    tensor, homeids = get_tensor(year, dataset)
    # for each fold, do the active learning with query by commitee
    for fold_num in range(5):
        print("Fold: ", fold_num)
        train, test, tr_homes, tt_homes = get_train_test(tensor, homeids, fold_num=fold_num)
        if init == 'random':
            season = None
        QBC[fold_num] = QueryByCommittee(train_tensor=train, test_tensor=test, init=init, reg=reg, pre_season=season,
                                         lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
                                         latent_dimension=latent_dimension, k=k, random_seed=random_seed)

        for t in range(12):
            QBC[fold_num].select(t)
            if season_type == 'fixed':
                QBC[fold_num].update_cm_fix_season(t)
            else:
                QBC[fold_num].update_cm_update_season(t)

        app_matrix[fold_num] = QBC[fold_num].app_matrix
        season_matrix[fold_num] = QBC[fold_num].season_matrix
        selected_pair[fold_num] = QBC[fold_num].selected_pair
        test_home_matrix[fold_num] = QBC[fold_num].test_home_matrix
        train_home_matrix[fold_num] = QBC[fold_num].train_home_matrix
        # std[fold_num] = AS[fold_num].uncertaint

    # get the prediction
    m, n, o = tensor.shape
    pred = np.empty((12, m, n, o))
    for t in range(12):
        # pred[t] = np.empty(tensor.shape)
        num_home = 0
        for fold_num in range(5):
            pr = np.einsum('Hr, Ar, Sr -> HAS', QBC[fold_num].test_home_matrix[t],
                           QBC[fold_num].app_matrix[t], QBC[fold_num].season_matrix[t])
            pred[t][num_home:(num_home + QBC[fold_num].test_home_matrix[t].shape[0])] = pr
            num_home += QBC[fold_num].test_home_matrix[t].shape[0]

    return pred, train_home_matrix, test_home_matrix, app_matrix, season_matrix, selected_pair


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", help="the year of the energy data", default=2015)
    parser.add_argument("--dataset", help="the type of dataset", default='artificial')
    parser.add_argument("--init", help="initialization method (pre, random)", default="pre")
    parser.add_argument("--k", help="number of selection at each trial", default=5)
    parser.add_argument("--latent_dimension", help="number of latent dimension", default=4)
    parser.add_argument("--season_type", help="updated or fixed season factor", default="updated")
    parser.add_argument("--regularization", help="true (new version) or false (previous version)", default="true")
    parser.add_argument("--lambda1", help="lambda1 for home factor", default=10000)
    parser.add_argument("--lambda2", help="lambda2 for appliance factor", default=10000)
    parser.add_argument("--lambda3", help="lambda3 for season factor", default=10000)

    args = parser.parse_args()
    print(args)
    # year, dataset, method, init, normalization, uncertainty, alpha1, alpha2, alpha3, k, latent_dimension, season_type, reg = sys.argv[1:]

    # initialize the parameters
    iters = 5
    lambda1 = int(args.lambda1)
    lambda2 = int(args.lambda2)
    lambda3 = int(args.lambda3)
    year = int(args.year)
    dataset = args.dataset
    init = args.init
    latent_dimension = int(args.latent_dimension)
    k = int(args.k)
    season_type = args.season_type
    reg = (args.regularization == "true")
    num_random = 10
    normalization = "none"

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
        pred, train_home_matrix, test_home_matrix, app_matrix, season_matrix, selected_pair = query_by_committee(year=year, dataset=dataset,
                                                                                                                 latent_dimension=latent_dimension,
                                                                                                                 lambda1=lambda1,
                                                                                                                 lambda2=lambda2,
                                                                                                                 lambda3=lambda3,
                                                                                                                 init=init,
                                                                                                                 reg=reg,
                                                                                                                 k=k,
                                                                                                                 random_seed=random_seed,
                                                                                                                 iters=iters,
                                                                                                                 season_type=season_type)
        prediction[random_seed] = pred
        season_m[random_seed] = season_matrix
        appliance_m[random_seed] = app_matrix
        sp[random_seed] = selected_pair
        train_h_m[random_seed] = train_home_matrix
        test_h_m[random_seed] = test_home_matrix

    # print("OK here")
    if reg:
        pre_dir = "../data/result/reg"
    else:
        pre_dir = "../data/result"

    if season_type == 'fixed':
        directory = "{}/fixed_season/qbc/{}/{}/".format(pre_dir, year, dataset)
    else:
        directory = "{}/update_season/qbc/{}/{}/".format(pre_dir, year, dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)

    pred_file = 'pred-{}-{}-{}-{}-{}-{}'.format(init, lambda1, lambda2, lambda3, k, latent_dimension)
    sp_file = 'sp-{}-{}-{}-{}-{}-{}'.format(init, lambda1, lambda2, lambda3, k, latent_dimension)
    am_file = 'app-matrix-{}-{}-{}-{}-{}-{}'.format(init, lambda1, lambda2, lambda3, k, latent_dimension)
    sm_file = 'season-matrix-{}-{}-{}-{}-{}-{}'.format(init, lambda1, lambda2, lambda3, k, latent_dimension)
    tr_file = 'train-matrix-{}-{}-{}-{}-{}-{}'.format(init, lambda1, lambda2, lambda3, k, latent_dimension)
    tt_file = 'test-matrix-{}-{}-{}-{}-{}-{}'.format(init, lambda1, lambda2, lambda3, k, latent_dimension)
    # ut_file = "uncertainty-{}-{}-{}-{}-{}-{}-{}-{}".format(init, normalization, uncertainty, alpha1,
    #                                                                alpha2, alpha3, k, latent_dimension)
    print(directory + pred_file)
    f = open(directory + pred_file, 'wb')
    pickle.dump(prediction, f)
    f.close()
    print(directory + am_file)
    f = open(directory + am_file, "wb")
    pickle.dump(app_matrix, f)
    f.close()
    print(directory + sm_file)
    f = open(directory + sm_file, "wb")
    pickle.dump(season_matrix, f)
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
    # print(directory + ut_file)
    # f = open(directory+ut_file, "wb")
    # pickle.dump(uncty, f)
    # f.close()
