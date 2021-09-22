import numpy as np

def absolute_orientation(X, Y):
    assert X.shape == Y.shape, 'Dimensions do not match'

    # discard nan values
    i = ~np.isnan(X)
    X = X[i].reshape(-1, 3).T
    Y = Y[i].reshape(-1, 3).T

    # center the clouds
    centroid_X = np.mean(X, axis=1, keepdims=True)
    centroid_Y = np.mean(Y, axis=1, keepdims=True)
    cX = np.matrix(X - centroid_X, copy=False)  # TODO replace np.matrix with np.ndarray
    cY = np.matrix(Y - centroid_Y, copy=False)

    # extract the parameters
    s = np.linalg.norm(cX[:, 0]) / np.linalg.norm(cY[:, 0])
    U, _, Vt = np.linalg.svd(cY * cX.T)
    D = np.diag([1, 1, np.linalg.det(U * Vt)])
    R = (U * D * Vt).T
    t = (1/s) * centroid_X - R * centroid_Y

    return s, R, t
