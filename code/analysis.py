import pickle


def get_baseline_results(directory, method='random', init='pre', lambda1=10000, lambda2=10000, lambda3=10000, k=5, latent_dimension=3):
    # load the prediction
    file_name = 'pred-{}-{}-{}-{}-{}-{}'.format(init, lambda1, lambda2, lambda3, k, latent_dimension)
    file = open(directory + file_name, 'rb')
    pred = pickle.load(file)
    # load the selected pairs
    # random_seed, fold_num, season_id
    file_name = 'sp-{}-{}-{}-{}-{}-{}'.format(init, lambda1, lambda2, lambda3, k, latent_dimension)
    file = open(directory + file_name, 'rb')
    sp = pickle.load(file)
    # load the appliance matrix
    # random_seed, fold_num, season_id
    file_name = 'app-matrix-{}-{}-{}-{}-{}-{}'.format(init, lambda1, lambda2, lambda3, k, latent_dimension)
    file = open(directory + file_name, 'rb')
    app_matrix = pickle.load(file)
    # load the home matrix
    # random_seed, fold_num, season_id
    file_name = 'test-matrix-{}-{}-{}-{}-{}-{}'.format(init, lambda1, lambda2, lambda3, k, latent_dimension)
    file = open(directory + file_name, 'rb')
    test_home = pickle.load(file)
    file_name = 'train-matrix-{}-{}-{}-{}-{}-{}'.format(init, lambda1, lambda2, lambda3, k, latent_dimension)
    file = open(directory + file_name, 'rb')
    train_home = pickle.load(file)
    # load the season matrix
    # random_seed, fold_num, season_id
    file_name = 'season-matrix-{}-{}-{}-{}-{}-{}'.format(init, lambda1, lambda2, lambda3, k, latent_dimension)
    file = open(directory + file_name, 'rb')
    season = pickle.load(file)

    result = {}
    result['prediction'] = pred
    result['sp'] = sp
    result['app_matrix'] = app_matrix
    result['test_home_matrix'] = test_home
    result['train_home_matrix'] = train_home
    result['season'] = season

    return result


def get_active_budget_results(year=2015, dataset='artificial', method='active', reg=True, season='updated',
                       alpha1=0.1, alpha2=0.1, alpha3=0.0, lambda1=10000, lambda2=10000, lambda3=10000, gamma1=0.5, gamma2=0.5, normalization='none',
                       uncertainty='current', kernel='gaussian', sigma=12, init='pre', k=55, latent_dimension=4):

    if reg:
        pre_dir = "../data/result/budget/reg"
    else:
        pre_dir = "../data/result/budget"

    if season == 'fixed':
        directory = "{}/fixed_season/{}/{}/{}/".format(pre_dir, method, year, dataset)
    else:
        directory = "{}/update_season/{}/{}/{}/".format(pre_dir, method, year, dataset)

    if method == 'random' or method == 'qbc':
        return get_baseline_results(directory=directory, method=method, init=init, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
                                    k=k, latent_dimension=latent_dimension)

    param_name = "{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(init, normalization, uncertainty,
                                                                 kernel, sigma, alpha1, alpha2, alpha3,
                                                                 lambda1, lambda2, lambda3, k, latent_dimension)

    e_file = "pred-{}".format(param_name)
    sp_file = "sp-{}".format(param_name)
    am_file = "app-matrix-{}".format(param_name)
    sm_file = "season-matrix-{}".format(param_name)
    tr_file = "train-matrix-{}".format(param_name)
    tt_file = "test-matrix-{}".format(param_name)
    # load prediction
    file = open(directory + e_file, 'rb')
    pred = pickle.load(file)
    file.close()

    file = open(directory + sp_file, 'rb')
    sp = pickle.load(file)
    file.close()
    # load the appliance matrix
    file = open(directory + am_file, 'rb')
    app_matrix = pickle.load(file)
    file.close()
    # load the home matrix
    # random_seed, fold_num, season_id
    file = open(directory + tt_file, 'rb')
    test_home = pickle.load(file)
    file.close()

    file = open(directory + tr_file, 'rb')
    train_home = pickle.load(file)
    file.close()
    # load the season matrix
    # random_seed, fold_num, season_id
    file = open(directory + sm_file, 'rb')
    season = pickle.load(file)
    file.close()

    result = {}
    result['prediction'] = pred
    result['sp'] = sp
    result['app_matrix'] = app_matrix
    result['test_home_matrix'] = test_home
    result['train_home_matrix'] = train_home
    result['season'] = season

    return result


def get_active_results(year=2015, dataset='artificial', method='active', reg=True, season='updated',
                       alpha1=0.1, alpha2=0.1, alpha3=0.0, lambda1=10000, lambda2=10000, lambda3=10000, gamma1=0.5, gamma2=0.5, normalization='none',
                       uncertainty='current', kernel='gaussian', sigma=12, init='pre', k=5, latent_dimension=4):

    if reg:
        pre_dir = "../data/result/reg"
    else:
        pre_dir = "../data/result"

    if season == 'fixed':
        directory = "{}/fixed_season/{}/{}/{}/".format(pre_dir, method, year, dataset)
    else:
        directory = "{}/update_season/{}/{}/{}/".format(pre_dir, method, year, dataset)

    if method == 'random' or method == 'qbc':
        return get_baseline_results(directory=directory, method=method, init=init, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
                                    k=k, latent_dimension=latent_dimension)

    param_name = "{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(init, normalization, uncertainty,
                                                                 kernel, sigma, alpha1, alpha2, alpha3,
                                                                 lambda1, lambda2, lambda3, k, latent_dimension)

    e_file = "pred-{}".format(param_name)
    sp_file = "sp-{}".format(param_name)
    am_file = "app-matrix-{}".format(param_name)
    sm_file = "season-matrix-{}".format(param_name)
    tr_file = "train-matrix-{}".format(param_name)
    tt_file = "test-matrix-{}".format(param_name)
    # load prediction
    file = open(directory + e_file, 'rb')
    pred = pickle.load(file)
    file.close()

    file = open(directory + sp_file, 'rb')
    sp = pickle.load(file)
    file.close()
    # load the appliance matrix
    file = open(directory + am_file, 'rb')
    app_matrix = pickle.load(file)
    file.close()
    # load the home matrix
    # random_seed, fold_num, season_id
    file = open(directory + tt_file, 'rb')
    test_home = pickle.load(file)
    file.close()

    file = open(directory + tr_file, 'rb')
    train_home = pickle.load(file)
    file.close()
    # load the season matrix
    # random_seed, fold_num, season_id
    file = open(directory + sm_file, 'rb')
    season = pickle.load(file)
    file.close()

    result = {}
    result['prediction'] = pred
    result['sp'] = sp
    result['app_matrix'] = app_matrix
    result['test_home_matrix'] = test_home
    result['train_home_matrix'] = train_home
    result['season'] = season

    return result
