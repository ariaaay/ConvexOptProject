import time
import numpy as np
import pickle
import spams
import sys
from gensim.models import KeyedVectors
from sklearn.decomposition import DictionaryLearning
import matplotlib.pyplot as plt

def test_2v2_accuracy():
    pass

def plot_rdm(X, Y, w):
    Xc = X[:w, :]
    Yc = Y[:w, :]
    plt.figure()
    plt.subplot(121)
    plt.imshow(np.corrcoef(Xc))
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(np.corrcoef(Yc))
    plt.colorbar()
    plt.show()

def avg_repeated_brain_trials(brain_data, brain_labels):
    """
    Average the duplicated trials in the brain data
    """
    brain_labels_unique = []
    label2trial = dict()
    for i, l in enumerate(brain_labels):
        if l not in brain_labels_unique:
            brain_labels_unique.append(l)
            label2trial[l] = [i]
        else:
            label2trial[l].append(i)
    
    brain_data_unique = np.zeros((len(brain_labels_unique), brain_data.shape[1]))
    for j, lab in enumerate(brain_labels_unique):
        if len(label2trial[lab]) == 1:
            brain_data_unique[j,:] = brain_data[label2trial[lab],:]
        else:
            brain_data_unique[j,:] = np.mean([brain_data[idx,:] for idx in label2trial[lab]])
    return brain_data_unique, brain_labels_unique

def extract_common_objs(brain_data, brain_labels, obj_vectors, obj_labels):
    """
    This functions takes in NON-repeated brain data.
    Outputs:
    - X: object embedding matrix of dimension w1-by-c. First w rows of X corresponds to objects
    that are also present in the brain data, in the same order as it is present in the brain data.
    - Xlabels: labels corresponding to the rows in X.
    - Y: the brain data matrix of dimension w2-by-v. 
    - Ylabels: labels corresponding to the rows in Y.
    - w: number of words that overlap
    """
    br_overlap_idx = []
    obj_overlap_idx = []

    for i, obj_lab in enumerate(obj_labels):
        for j, br_lab in enumerate(brain_labels):
            if obj_lab == br_lab:
                obj_overlap_idx.append(i)
                br_overlap_idx.append(j)
    assert(len(obj_overlap_idx) == len(br_overlap_idx))
    w = len(obj_overlap_idx)

    br_nonoverlap_idx = np.setdiff1d(np.arange(len(brain_labels)), br_overlap_idx)
    obj_nonoverlap_idx = np.setdiff1d(np.arange(len(obj_labels)), obj_overlap_idx)

    X = np.vstack((obj_vectors[obj_overlap_idx,:], obj_vectors[obj_nonoverlap_idx,:]))
    Y = np.vstack((brain_data[br_overlap_idx,:], brain_data[br_nonoverlap_idx,:]))

    return X, Y, w

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

def main(X, Y, w, K=100, lamb=0.025):
    model_X, D_Y = None, None
    params = {'n_components':K, 'alpha':lamb, 'max_iter': 10, 'n_jobs': -1, \
    'positive_code':True, 'transform_algorithm':'lasso_lars'}
    t = 0
    loss_X_arr = []
    loss_Y_arr = []
    while t < 3:
        if model_X is None:
            model_X = DictionaryLearning(**params)
        else:
            A_X[w,:] = A_Y[w,:]
            model_X = DictionaryLearning(code_init=A_X, dict_init=D_X, **params)
        
        model_X.fit(X)
        A_X = model_X.transform(X)
        D_X = model_X.components_

        reconstruction_X = A_X @ D_X
        xd = X - reconstruction_X
        loss_X = np.mean((xd**2).sum(axis=1) + lamb * np.abs(A_X).sum(axis=1))
        loss_X_arr.append(loss_X)
        print('Loss of X: %f' % loss_X)

        #warm start alpha in model_Y
        A_Y = np.zeros((Y.shape[0], K))
        A_Y[w,:] = A_X[w,:]
        if D_Y is None:
            model_Y = DictionaryLearning(code_init=A_Y, **params)
        else:
            model_Y = DictionaryLearning(code_init=A_Y, dict_init=D_Y, **params)

        model_Y.fit(Y)
        A_Y = model_Y.transform(Y)
        D_Y = model_Y.components_

        reconstruction_Y = A_Y @ D_Y
        yd = Y - reconstruction_Y
        loss_Y = np.mean((yd**2).sum(axis=1) + lamb * np.abs(A_Y).sum(axis=1))
        loss_Y_arr.append(loss_Y)
        print('Loss of Y: %f' % loss_Y)

        t+=1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(t), loss_Y_arr)
    ax2 = ax.twinx()
    ax2.plot(np.arange(t), loss_X_arr)
    plt.show()

    return D_X, D_Y, A_Y, A_X

if __name__ == '__main__':
    try:
        brain_data_path = sys.argv[1]
        brain_labels_path = sys.argv[2]
        obj_embedding_path = sys.argv[3]
    except IndexError:
        brain_data_path = "./data/S1_PPA_LH.npy"
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
    brain_data_unique, brain_labels_unique = avg_repeated_brain_trials(brain_data, brain_labels)
    X, Y, w = extract_common_objs(brain_data_unique, brain_labels_unique, obj_vectors, obj_labels)
    # print("Linear project residual is: " +str(linear_test(X, Y)))


    # D_X, D_Y, A_Y, A_X = main(X, Y, w)

