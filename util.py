import numpy as np

def takeout_repeated_brain_trials(brain_data, brain_labels):
    """
    Take out the duplicated trials in the brain data
    """
    brain_labels_unique = []
    label2trial = dict()
    for i, l in enumerate(brain_labels):
        if l not in brain_labels_unique:
            brain_labels_unique.append(l)
            label2trial[l] = i
    # print(label2trial)

    brain_data_unique = np.zeros((len(brain_labels_unique), brain_data.shape[1]))
    for j, lab in enumerate(brain_labels_unique):
        brain_data_unique[j, :] = brain_data[label2trial[lab], :]
    return brain_data_unique, brain_labels_unique

# def avg_repeated_brain_trials(brain_data, brain_labels):
#     """
#     Average the duplicated trials in the brain data
#     """
#     brain_labels_unique = []
#     label2trial = dict()
#     for i, l in enumerate(brain_labels):
#         if l not in brain_labels_unique:
#             brain_labels_unique.append(l)
#             label2trial[l] = [i]
#         else:
#             label2trial[l].append(i)
#     # print(label2trial)
#
#     brain_data_unique = np.zeros((len(brain_labels_unique), brain_data.shape[1]))
#     for j, lab in enumerate(brain_labels_unique):
#         if len(label2trial[lab]) == 1:
#             brain_data_unique[j,:] = brain_data[label2trial[lab],:]
#         else:
#             brain_data_unique[j,:] = np.mean([brain_data[idx,:] for idx in label2trial[lab]])
#     return brain_data_unique, brain_labels_unique

def simulate_data(w0, w1, w2, l=100, c=200):
    numel = l*(w0+w1+w2)
    #choose 80% sparsity
    A = np.random.rand(1,numel)
    idx = np.random.choice(range(numel),size=np.int(np.round(.8*numel)),replace=False)
    #A = A.reshape(1,-1)
    A[0,idx] = 0
    A = A.reshape(w0+w1+w2, l)

    D_X = np.random.rand(l, c)
    D_Y = np.random.rand(l, c)
    X = A[:(w0+w1), :] @ D_X
    Y = np.vstack((A[:w0, :], A[w0+w1:,:])) @ D_Y
    return X, Y, A[:(w0+w1),:]


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