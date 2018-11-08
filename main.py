import numpy as np
import pickle
import spams
import sys
from gensim.models import KeyedVectors

def test_2v2_accuracy():
    pass

def main(brain_data, brain_labels, obj_vectors, obj_labels, K=200):
    # TODO: Change the SPAMS settings to make sure it's being run with the 
    #       constraints and penalties that are described in the paper.
    print(brain_data.shape) 
    print(obj_vectors.shape)
    # First, run a simple baseline.
    # (1) Train a dictionary learning model just on the object embeddings
    #     and measure 2v2 accuracy using an arbitrary brain data file.
    X = np.asfortranarray(obj_vectors)
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
    # Read only the object rows that correspond to the brain labels.
    # Train a model XW = A using L2-regularized linear regression.
    # Use trained W to compute 2v2 accuracy on a holdout set.

if __name__ == '__main__':
    brain_data_path = sys.argv[1]
    brain_labels_path = sys.argv[2]
    obj_embedding_path = sys.argv[3]

    # Brain data is provided as a single numpy array, labels as a pickled
    # Python list.
    brain_data = np.load(brain_data_path)
    brain_labels = pickle.load(open(brain_labels_path, 'rb'))
    # Object embeddings are read from a gensim model file.
    wv_model = KeyedVectors.load(obj_embedding_path, mmap='r')
    pix2vec = wv_model.vectors
    wv_list = list(wv_model.vocab)

    main(brain_data, brain_labels, pix2vec, wv_list)
