# -*-coding:utf-8-*-
import scipy.io
from sparseNMF import sparseNMF
import matplotlib.pyplot as plt


def main():

    inp_data = scipy.io.loadmat('imagedata.mat')
    v = inp_data['f']

    # params
    r = 500  # number of atoms
    cf = 'kl'  # 'is', 'kl', 'ed'
    sparsity = 5
    # stopping criteria
    max_iter = 100
    conv_eps = 1e-3
    display = True
    random_seed = 1

    w, h, objective_div, objective_cost = sparseNMF(
        v, cf=cf, sparsity=sparsity, max_iter=max_iter, conv_eps=conv_eps,
        display=display, random_seed=random_seed, r=r)

    fig, ax = plt.subplots(1, 4, figsize=[16, 4], constrained_layout=True)
    cmap = plt.get_cmap('jet')
    ax[0].imshow(v, aspect='auto', cmap=cmap)
    ax[0].set_title('input')
    ax[1].plot(objective_div, label='divergence')
    ax[1].plot(objective_cost, label='cost')
    ax[1].set_title('cost')
    ax[2].imshow(w, aspect='auto', cmap=cmap)
    ax[2].set_title('W')
    ax[3].imshow(h, aspect='auto', cmap=cmap)
    ax[3].set_title('H')
    fig.savefig('images/eg_python.png')


if __name__ == '__main__':
    main()
