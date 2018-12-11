import time
import numpy as np
from tqdm import tqdm
import pickle
# import spams
import sys
from gensim.models import KeyedVectors
from sklearn.decomposition import DictionaryLearning
from sklearn.cluster.bicluster import SpectralBiclustering
import matplotlib.pyplot as plt
from util import *

def test_2v2_accuracy():
    pass

def plot_rdm(X, Y, w):
    Xc = X[:w, :]
    Yc = Y[:w, :]
    plt.figure()

    Xcorr = np.corrcoef(Xc)
    Ycorr = np.corrcoef(Yc)

    model = SpectralBiclustering(n_clusters=20, method='log', random_state=0)
    model.fit(Ycorr)
    fitX = Xcorr[np.argsort(model.row_labels_)]
    fitX = fitX[:, np.argsort(model.column_labels_)]

    fitY = Ycorr[np.argsort(model.row_labels_)]
    fitY = fitY[:, np.argsort(model.column_labels_)]
    plt.subplot(121)
    plt.imshow(fitX, cmap=plt.cm.Blues)
    plt.title('Image Object Space')
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(fitY, cmap=plt.cm.Blues)
    plt.title('Brain Responses')
    plt.colorbar()
    plt.show()





# def main(X, Y, w, K=100):
#     # TODO: Change the SPAMS settings to make sure it's being run with the 
#     #       constraints and penalties that are described in the paper.
#     return
#     #alternate optimization
#     X = np.asfortranarray(X.T)
#     Y = np.asfortranarray(Y.T)

#     params = {'K' : K, 'lambda1' : 0.025, 'numThreads' : 32,'iter' : 1}
#     lasso_params = {'lambda1': 0.025, 'numThreads' : 32}


#     D_X, model_X = spams.trainDL(X, return_model=True, batch=True, **params)
#     # print(D_X.shape)
#     # print(model_X['A'].shape)
#     print(model_X['B'].shape)
    
#     alpha = spams.lasso(X, D=D_X, **lasso_params)
#     print(alpha.shape)
#     reconstruction_X = D_X * alpha
#     xd = X - reconstruction_X
#     loss_X = np.mean(0.5 * (xd * xd).sum(axis=0) + params['lambda1'] * np.abs(alpha).sum(axis=0))
#     print('Loss of X: %f' % loss_X)

#     # D_Y, model_Y = spams.trainDL(X, return_model=True, **params)

#     # Convert alpha back to dense.
#     alpha = alpha.toarray()
#     # Train a model XW = A using L2-regularized linear regression.
#     # Use trained W to compute 2v2 accuracy on a holdout set.

def joint_optimize(X, Y, w, w1, fit_algo, transform_algo, K=100, lamb=0.1):
    model_X, D_Y = None, None
    params = {'n_components':K, 'alpha':lamb, 'max_iter': 100, 'n_jobs': -1, 'positive_code':True,
              'transform_algorithm':transform_algo, 'fit_algorithm':fit_algo, 'tol':1e-02}
    t = 0
    loss_X_arr = []
    loss_Y_arr = []
    converged = False
    tol = 1e-3
    curr_obj = np.inf
    while not converged and t < 300:
        prev_obj = curr_obj

        if model_X is None:
            model_X = DictionaryLearning(**params)
            model_joint = DictionaryLearning(**params)
            model_Y = DictionaryLearning(**params)
        else:
            model_X = DictionaryLearning(code_init=A_X, dict_init=D_X, **params)
            model_Y = DictionaryLearning(code_init=A_Y, dict_init=D_Y, **params)
            model_joint = DictionaryLearning(code_init=A, dict_init = np.hstack((D_X, D_Y)), **params)
        
        model_X.fit(X)
        A_X = model_X.transform(X)
        D_X = model_X.components_

        model_Y.fit(Y)
        A_Y = model_Y.transform(Y)
        D_Y = model_Y.components_

        model_joint.fit(np.hstack((X[:w,:], Y[:w,:])))
        A_joint = model_joint.transform(np.hstack((X[:w,:], Y[:w,:])))

        A = np.zeros((w+w1+w1, K)) #only in simulation data

        A[:w,:] = A_joint
        A[w:w+w1,:] = A_X[w:,:]
        A[w+w1:,:] = A_Y[w:,:]
        A_X[:w,:] = A[:w,:]
        A_Y[:w,:] = A[:w,:]

        loss_X = eval(X, A[:w+w1,:], D_X)
        loss_X_arr.append(loss_X)
        loss_Y = eval(Y, np.vstack((A[:w,:], A[w+w1:,:])), D_Y)
        loss_Y_arr.append(loss_Y)

        np.save("./outputs/joint_loss_X_{}_{}_{}.npy".format(transform_algo, fit_algo, lamb), loss_X_arr)
        np.save("./outputs/joint_loss_Y_{}_{}_{}.npy".format(transform_algo, fit_algo, lamb), loss_Y_arr)

        t+=1
        curr_obj = loss_X + loss_Y
        converged = np.abs(prev_obj - curr_obj) <= tol



    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(t), loss_Y_arr)
    ax2 = ax.twinx()
    ax2.plot(np.arange(t), loss_X_arr)
    plt.savefig("./figures/joint_{}_{}_{}.png".format(transform_algo, fit_algo, lamb))

    return D_X, D_Y, A

def main_optimize(X, Y, w, fit_algo, transform_algo, K=100, lamb=0.025):
    model_X, D_Y = None, None
    params = {'n_components': K, 'alpha': lamb, 'max_iter': 500, 'n_jobs': -1, 'positive_code': True,
              'transform_algorithm': transform_algo, 'fit_algorithm': fit_algo, 'tol': 1e-02}
    t = 0
    loss_X_arr = []
    loss_Y_arr = []
    converged = False
    tol = 1e-3
    curr_obj = np.inf
    while not converged and t < 300:
        prev_obj = curr_obj

        if model_X is None:
            model_X = DictionaryLearning(**params)
        else:
            A_X[:w, :] = A_Y[:w, :]
            model_X = DictionaryLearning(code_init=A_X, dict_init=D_X, **params)

        model_X.fit(X)
        A_X = model_X.transform(X)
        D_X = model_X.components_
        loss_X = eval(X, A_X, D_X)
        loss_X_arr.append(loss_X)
        # print('Loss of X: %f' % loss_X)

        # warm start alpha in model_Y
        A_Y = np.zeros((Y.shape[0], K))
        A_Y[:w, :] = A_X[:w, :]
        if D_Y is None:
            model_Y = DictionaryLearning(code_init=A_Y, **params)
        else:
            model_Y = DictionaryLearning(code_init=A_Y, dict_init=D_Y, **params)

        model_Y.fit(Y)
        A_Y = model_Y.transform(Y)
        D_Y = model_Y.components_

        loss_Y = eval(Y, A_Y, D_Y)
        loss_Y_arr.append(loss_Y)
        # print('Loss of Y: %f' % loss_Y)

        t += 1
        curr_obj = loss_X + loss_Y
        converged = np.abs(prev_obj - curr_obj) <= tol
        np.save("./outputs/loss_X_{}_{}_{}.npy".format(transform_algo, fit_algo, lamb), loss_X_arr)
        np.save("./outputs/loss_Y_{}_{}_{}.npy".format(transform_algo, fit_algo, lamb), loss_Y_arr)

    # print(loss_X_arr)
    # print(loss_Y_arr)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(t), loss_Y_arr)
    ax2 = ax.twinx()
    ax2.plot(np.arange(t), loss_X_arr)
    plt.savefig("./figures/{}_{}.png".format(transform_algo, fit_algo))

    return D_X, D_Y, A_Y, A_X


def eval(X, A, D, lamb=0):
    recon_X = A @ D
    diff = X-recon_X
    loss = np.mean((diff**2).sum(axis=1) + lamb * np.abs(A).sum(axis=1))
    return loss


if __name__ == '__main__':
    try:
        brain_data_path = sys.argv[2]
        brain_labels_path = sys.argv[3]
        obj_embedding_path = sys.argv[4]
    except IndexError:
        brain_data_path = "./data/S1_LOC_LH.npy"
        brain_labels_path = "./data/image_category.p"
        obj_embedding_path = "./data/pix2vec_200.model"


    # Brain data is provided as a single numpy array, labels as a pickled
    # Python list
    brain_data = np.load(brain_data_path)
    brain_labels = pickle.load(open(brain_labels_path, 'rb'))
    # Object embeddings are read from a gensim model file.
    wv_model = KeyedVectors.load(obj_embedding_path, mmap='r')
    obj_vectors = wv_model.vectors
    obj_labels = list(wv_model.vocab)
    brain_data_unique, brain_labels_unique = takeout_repeated_brain_trials(brain_data, brain_labels)
    X, Y, w = extract_common_objs(brain_data_unique, brain_labels_unique, obj_vectors, obj_labels)
    # print("Linear project residual is: " +str(linear_test(X, Y)))
    # plot_rdm(X, Y, w)
    w0, w1, w2 = 200, 100, 100
    Xsim, Ysim, Asim, Dsimx, Dsimy = simulate_data(w0, w1, w2, return_D=True)
    np.save("Xsim.npy", Xsim)
    np.save("Ysim.npy", Ysim)
    np.save("Asim.npy", Asim)
    # Ax = Asim[:w0+w1,:]
    # Ay = np.vstack((Asim[:w0, :], Asim[w0+w1:,:]))
    # sim_loss = eval(Xsim, Ax, Dsimx) + eval(Ysim, Ay, Dsimy)

    # plot_rdm(Xsim, Ysim, w)

    transform_algorithm = ['lasso_lars', 'lasso_cd']
    fit_algorithm = ['lars', 'cd']
    lambs = np.logspace(-2, 1, 4)
    #
    # transform_algorithm = ['lasso_lars']
    # fit_algorithm = ['lars']
    for tr in tqdm(transform_algorithm):
        for ft in tqdm(fit_algorithm):
            for la in tqdm(lambs):
                # D_X, D_Y, A_Y, A_X = main(X, Y, w)
                print("testing with {} and {}".format(ft, tr))
                if sys.argv[1] == "simulation":
                    D_X, D_Y, A_Y, A_X = main_optimize(Xsim, Ysim, w0, ft, tr, lamb=la)
                    D_X, D_Y, A = joint_optimize(Xsim, Ysim, w0, w1, ft, tr, lamb=la)
                else:
                    D_X, D_Y, A_Y, A_X = main_optimize(X, Y, w0, ft, tr, lamb=la)
                    D_X, D_Y, A = joint_optimize(X, Y, w0, w1, ft, tr, lamb=la)


