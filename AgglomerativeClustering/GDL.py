import numpy as np
from affinity import *



def GDL(X, nt, Kc):

    Vc = k0graph(X)
    nc = len(Vc)

    print("AgglomerativeClustering")
    while nc > nt:
        A = Affinity(X, Vc, Kc)
        print("Affinity Complete")
        m = np.unravel_index(np.argmax(A), A.shape)
        Vc[m[0]].extend(Vc[m[1]])
        Vc.pop(m[1])
        nc = nc - 1
        print(nc)

    return Vc


if __name__ == '__main__':
    sample = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [3, 4], [1, 7], [5, 5], [5, 6], [6, 7], [7, 7]])
    Vc = GDL(sample,2, 3)
    print(Vc)

