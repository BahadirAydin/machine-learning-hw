import numpy as np
import matplotlib.pyplot as plt

# this function has been adapted from:
# https://scipython.com/blog/plotting-the-decision-boundary-of-a-logistic-regression-model/
# http://www.cse.chalmers.se/~richajo/dit866/backup_2019/lectures/l3/Plotting%20decision%20boundaries.html

plt.ion()
def display_boundary(X, W, b, label):
    plt.clf()
    h = .01  # step size in the mesh, we can decrease this value for smooth plots, i.e 0.01 (but ploting may slow down)
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 3, X[:, 0].max() + 3
    y_min, y_max = X[:, 1].min() - 3, X[:, 1].max() + 3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    aa = np.c_[xx.ravel(), yy.ravel()]
    Z = np.sign(np.matmul(aa, W)+b).reshape(-1)
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
    plt.contourf(xx, yy, Z, colors=['red', 'green'],  alpha=0.25) # cmap="Paired_r",
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=label, cmap="Paired_r", edgecolors='k');
    x_ = np.array([x_min, x_max])
    m = -W[0]/W[1]
    c = -b[0]/W[1]
    y_ = m * x_ + c
    plt.plot(x_, y_, 'k', lw=4, ls='-')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.title("W=(%.4f,%.4f), b=%.4f"% (W[0], W[1],b[0]))
    plt.pause(0.001)  # <---- add pause
    plt.show()