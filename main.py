import time
import numpy as np
import pickle
import spams
import sys
from gensim.models import KeyedVectors

def test_2v2_accuracy():
    pass

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
    """
    br_overlap_idx = []
    obj_overlap_idx = []

    for i, obj_lab in enumerate(obj_labels):
        for j, br_lab in enumerate(brain_labels):
            if obj_lab == br_lab:
                obj_overlap_idx.append(i)
                br_overlap_idx.append(j)

    br_nonoverlap_idx = np.setdiff1d(np.arange(len(brain_labels)), br_overlap_idx)
    obj_nonoverlap_idx = np.setdiff1d(np.arange(len(obj_labels)), obj_overlap_idx)

    X = np.vstack((brain_data[br_overlap_idx,:], brain_data[br_nonoverlap_idx,:]))
    Y = np.vstack((obj_vectors[obj_overlap_idx,:], obj_vectors[obj_nonoverlap_idx,:]))

    return X, Y

def main(X, Y, K=200):
    # TODO: Change the SPAMS settings to make sure it's being run with the 
    #       constraints and penalties that are described in the paper.

    # First, run a simple baseline.
    # (1) Train a dictionary learning model just on the object embeddings
    #     and measure 2v2 accuracy using an arbitrary brain data file.

    #alternate optimization
    X = np.asfortranarray(X)
    params = {'K' : K, 'lambda1' : 0.025, 'numThreads' : 32,
              'batchsize' : 400, 'iter' : 50}

    tic = time.time()
    D = spams.trainDL_Memory(X, **params)
    tac = time.time()
    t = tac - tic
    print('time of computation for Dictionary Learning: %f' %t)
    lasso_params = {'lambda1': 0.025, 'numThreads' : 32}
    alpha = spams.lasso(X, D=D, **lasso_params)
    reconstruction = D * alpha
    xd = X - reconstruction
    loss = np.mean(0.5 * (xd * xd).sum(axis=0) + params['lambda1'] * np.abs(alpha).sum(axis=0))
    print('Loss: %f' % loss)
    # Convert alpha back to dense.
    alpha = alpha.toarray()
    # Train a model XW = A using L2-regularized linear regression.
    # Use trained W to compute 2v2 accuracy on a holdout set.

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
    X, Y = extract_common_objs(brain_data_unique, brain_labels_unique, obj_vectors, obj_labels)

    main(X, Y)
