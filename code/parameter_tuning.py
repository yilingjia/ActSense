import numpy as np
from analysis import *
from basic import *
from sklearn.metrics import mean_squared_error
import pandas as pd
import argparse


def active(year=2015, dataset='artificial', uncertainty='prev_future_weighted'):

    result = {}
    c = 0
    param = {}
    for k in [5]:
        for latent_dimension in [4]:
            for lambda1 in [5000, 8000, 10000]:
                for lambda2 in [5000, 8000, 10000]:
                    for lambda3 in [5000, 8000, 10000]:
                        for kernel in ['gaussian', 'triangle', 'cosine', 'circle']:
                            for sigma in [1, 3, 6, 12]:
                                result[c] = get_active_results(year=year, dataset=dataset, k=k,
                                                               uncertainty=uncertainty,
                                                               latent_dimension=latent_dimension, lambda1=lambda1,
                                                               lambda2=lambda2, lambda3=lambda3, kernel=kernel,
                                                               sigma=sigma)['prediction']
                                param[c] = '{}-{}-{}-{}-{}-{}-{}'.format(k, latent_dimension, lambda1, lambda2, lambda3,
                                                                         kernel, sigma)
                                c += 1

    return result, param


def random_qbc(year=2015, method='random', dataset='artificial'):

    result = {}
    c = 0
    param = {}
    for k in [5]:
        for latent_dimension in [3, 4]:
            for lambda1 in [5000, 8000, 10000]:
                for lambda2 in [5000, 8000, 10000]:
                    for lambda3 in [5000, 8000, 10000]:
                        result[c] = get_active_results(year=year, dataset=dataset, k=k, method=method,
                                                       latent_dimension=latent_dimension, lambda1=lambda1,
                                                       lambda2=lambda2, lambda3=lambda3)['prediction']
                        param[c] = '{}-{}-{}-{}-{}'.format(k, latent_dimension, lambda1, lambda2, lambda3)
                        c += 1
    return result, param


def vb(year=2015, dataset='artificial', method='vb_var'):
    c = 0
    result = {}
    param = {}
    for k in [5]:
        for latent_dimension in [4]:
            for a_gamma0 in [1.0, 0.1, 0.01]:
                for b_gamma0 in [1.0, 0.1, 0.01]:
                    for a_beta0 in [1.0, 0.1, 0.01]:
                        for b_beta0 in [1.0, 0.1, 0.01]:
                            for a_alpha0 in [1.0, 0.1, 0.01]:
                                for b_alpha0 in [1.0, 0.1, 0.01]:
                                    # pre_dir = "../data/result/vb_var/{}/artificial".format(year)
                                    #
                                    param_name = "{}-{}-{}-{}-{}-{}-{}-{}".format(k, latent_dimension, a_gamma0,
                                                                                  b_gamma0, a_beta0, b_beta0, a_alpha0,
                                                                                  b_alpha0)
                                    #
                                    # full_path = "{}/pred-{}".format(pre_dir, param_name)
                                    # print(full_path)
                                    # my_file = Path(full_path)
                                    # if not my_file.exists():
                                    #
                                    #     CMD = 'python vb_var_new.py --year {} --k {} --latent_dimension {}' \
                                    #           ' --a_gamma0 {} --b_gamma0 {} --a_beta0 {} --b_beta0 {} --a_alpha0 {}' \
                                    #           ' --b_alpha0 {}'.format(year, k, latent_dimension, a_gamma0, b_gamma0,
                                    #                                   a_beta0, b_beta0, a_alpha0, b_alpha0)
                                    #     print(CMD)
                                    #     cmds[c] = CMD
                                    #     c += 1
                                    result[c] = np.load("../data/result/{}/{}/{}/pred-{}".format(method, year, dataset,
                                                                                                 param_name))

                                    param[c] = '{}-{}-{}-{}-{}-{}-{}-{}'.format(k, latent_dimension, a_gamma0, b_gamma0,
                                                                                a_beta0, b_beta0, a_alpha0, b_alpha0)
                                    c += 1
    return result, param


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", help="the year of the energy data", default=2017)
    parser.add_argument("--dataset", help="the type of dataset", default='artificial')
    parser.add_argument("--method", help="method: active, random, qbc, vb", default='vb')
    parser.add_argument("--uncertainty", default='prev_future_weighted')

    args = parser.parse_args()
    print(args)

    order = APPLIANCE_ORDER_OBSERVED
    year = int(args.year)
    dataset = args.dataset
    uncertainty = args.uncertainty
    method = args.method

    # load the results

    result = {}
    c = 0

    if method == 'active':
        result, param = active(year, dataset, uncertainty)
    if method == 'random' or method == 'qbc':
        result, param = random_qbc(year, method, dataset)
    if method == 'vb_var' or method == 'vb_var_new':
        result, param = vb(year, dataset)

    tensor, homeidx = get_tensor(year, dataset)

    if method != 'vb_var' and method != 'vb_var_new':

        error = {}
        for idx, p in enumerate(param):
            error[p] = {}
            for idx, appliance in enumerate(order[1:]):
                error[p][appliance] = {}
                for t in range(12):
                    error[p][appliance][t] = {}
                    for random_seed in range(10):
                        error[p][appliance][t][random_seed] = {}

        for c, p in enumerate(param):
            for t in range(12):
                for idx, appliance in enumerate(order[1:]):
                    idx += 1
                    for random_seed in range(10):
                        predApp = result[c][random_seed][t]
                        error[p][appliance][t][random_seed] = np.sqrt(mean_squared_error(predApp[:, idx, t],
                                                                                         tensor[:, idx, t]))

        tmp = {}
        mean_error = {}
        for c, p in enumerate(param):
            tmp[p] = {}
            for idx, appliance in enumerate(order[1:]):
                tmp[p][appliance] = pd.DataFrame(error[p][appliance]).mean(axis=0).mean(axis=0)
            mean_error[p] = pd.Series(tmp[p]).mean()

        best_index = pd.Series(mean_error).argmin()
        print('parameter: ', param[best_index])

        best_app_error = {}
        for idx, appliance in enumerate(order[1:]):
            best_app_error[appliance] = pd.DataFrame(error[best_index][appliance]).mean(axis=0)
    else:
        error = {}
        mean_error = {}
        for c, p in enumerate(param):
            error[p] = {}
            for idx, appliance in enumerate(order[1:]):
                idx += 1
                error[p][appliance] = {}
                for t in range(12):
                    e = 0
                    for random_seed in range(5):
                        pr = result[p][random_seed][t][:, idx, t]
                        gt = tensor[:, idx, t]
                        e += np.sqrt(mean_squared_error(pr, gt))
                    error[p][appliance][t] = e / 5
        for c, p in enumerate(param):
            mean_error[p] = pd.DataFrame(error[p]).mean(axis=0).mean(axis=0)

        best_index = pd.Series(mean_error).argmin()
        print(pd.Series(mean_error))
        print('parameter: ', param[best_index], pd.Series(mean_error).min())

        best_app_error = {}
        for idx, appliance in enumerate(order[1:]):
            best_app_error[appliance] = error[best_index][appliance]
            print(appliance)
            print(best_app_error[appliance])

    if method == 'active':
        np.save("../data/best-param-{}-{}-{}".format(year, method, uncertainty), param[best_index])
        np.save("../data/best-result-{}-{}-{}".format(year, method, uncertainty), best_app_error)
    else:
        np.save("../data/best-param-{}-{}".format(year, method), param[best_index])
        np.save("../data/best-result-{}-{}".format(year, method), best_app_error)

    print("Done")
