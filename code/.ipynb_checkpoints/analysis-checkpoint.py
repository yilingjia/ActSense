import pickle


def get_baseline_results(directory, method='random', init='pre', k=5, latent_dimension=3):
    # load the prediction
    file_name = 'pred-{}-{}-{}'.format(init, k, latent_dimension)
    file = open(directory + file_name, 'rb')
    pred = pickle.load(file)
    # load the selected pairs
    # random_seed, fold_num, season_id
    file_name = 'sp-{}-{}-{}'.format(init, k, latent_dimension)
    file = open(directory + file_name, 'rb')
    sp = pickle.load(file)
    # load the appliance matrix
    # random_seed, fold_num, season_id
    file_name = 'app-matrix-{}-{}-{}'.format(init, k, latent_dimension)
    file = open(directory + file_name, 'rb')
    app_matrix = pickle.load(file)
    # load the home matrix
    # random_seed, fold_num, season_id
    file_name = 'test-matrix-{}-{}-{}'.format(init, k, latent_dimension)
    file = open(directory + file_name, 'rb')
    test_home = pickle.load(file)
    file_name = 'train-matrix-{}-{}-{}'.format(init, k, latent_dimension)
    file = open(directory + file_name, 'rb')
    train_home = pickle.load(file)
    # load the season matrix
    # random_seed, fold_num, season_id
    file_name = 'season-matrix-{}-{}-{}'.format(init, k, latent_dimension)
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


def get_active_results(year=2015, dataset='artificial', method='active', reg=False, season='fixed', 
                       alpha1=0.1, alpha2=0.1, alpha3=0.0, normalization='none', uncertainty='one', init='pre', k=5, latent_dimension=3):

    if reg:
        pre_dir = "../data/reg"
    else:
        pre_dir = "../data"

    if season == 'fixed':
        directory = "{}/fixed_season/{}/{}/{}/".format(pre_dir, method, year, dataset)
    else:
        directory = "{}/update_season/{}/{}/{}/".format(pre_dir, method, year, dataset)

    if method == 'random' or method == 'qbc':
        return get_baseline_results(directory=directory, method=method, init=init, k=k, latent_dimension=latent_dimension) 

    # load prediction
    file_name = "pred-{}-{}-{}-{}-{}-{}-{}-{}".format(init, normalization, uncertainty,alpha1, alpha2, alpha3, k, latent_dimension)
    file = open(directory + file_name, 'rb')
    pred = pickle.load(file)
    file.close()

    file_name = "sp-{}-{}-{}-{}-{}-{}-{}-{}".format(init, normalization, uncertainty,alpha1, alpha2, alpha3, k, latent_dimension)
    file = open(directory + file_name, 'rb')
    sp = pickle.load(file)
    file.close()
    # load the appliance matrix
    file_name = "app-matrix-{}-{}-{}-{}-{}-{}-{}-{}".format(init, normalization, uncertainty,alpha1, alpha2, alpha3, k, latent_dimension)
    file = open(directory + file_name, 'rb')
    app_matrix = pickle.load(file)
    file.close()
    # load the home matrix
    # random_seed, fold_num, season_id
    file_name = "test-matrix-{}-{}-{}-{}-{}-{}-{}-{}".format(init, normalization, uncertainty,alpha1, alpha2, alpha3, k, latent_dimension)
    file = open(directory + file_name, 'rb')
    test_home = pickle.load(file)
    file.close()
    file_name = "train-matrix-{}-{}-{}-{}-{}-{}-{}-{}".format(init, normalization, uncertainty,alpha1, alpha2, alpha3, k, latent_dimension)
    file = open(directory + file_name, 'rb')
    train_home = pickle.load(file)
    file.close()
    # load the season matrix
    # random_seed, fold_num, season_id
    file_name = "season-matrix-{}-{}-{}-{}-{}-{}-{}-{}".format(init, normalization, uncertainty, alpha1, alpha2, alpha3, k, latent_dimension)
    file = open(directory + file_name, 'rb')
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
