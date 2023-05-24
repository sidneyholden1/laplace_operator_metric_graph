"""
This script requires the sphere_m_ops branch of Dedalus3: https://dedalus-project.readthedocs.io/en/latest/
"""

import numpy as np
import scipy as sp
import sys

from graph_modules.modules import *

calculate_data = sys.argv[1]

if calculate_data == "True":

    from dedalus.libraries.dedalus_sphere import sphere as sphere
    from dedalus.libraries.dedalus_sphere.operators import infinite_csr

    def S2(p,k):

        sign = lambda k: k // (abs(k) + (k==0))

        a, na = p*((k==0) + sign(k)), abs(k + 1)
        b, nb = p*((k==0) - sign(k)), abs(k - 1)

        A = sphere.operator('Id-Cos')
        B = sphere.operator('Id+Cos')

        W = A(a)**na @ B(b)**nb

        I = sphere.operator('Id')
        Z = sphere.operator('Cos')

        if k < 0:
            Z *= -1

        phi = ( 1 + 5**(1/2) ) / 2

        if abs(k) == 0:
            P = (15/16) * (I - 18*Z**2 + 33*Z**4)
            P *= (-2726 +  664 * phi) / 7680

        if abs(k) == 1:
            P = (1/16) * (-17*I - 108*Z + 90*Z**2 + 660*Z**3 + 495*Z**4)
            P *= ( 2027 - 2394 * phi) / 5120

        if abs(k) == 2:
            P = (1/2) * (I + 22*Z + 33*Z**2)
            P *= ( 1363 -  332 * phi) / 4096

        if abs(k) == 3:
            P = I
            P *= (-6081 + 7182 * phi) / 16384

        return ((17 - 9 * phi) / 27) * (P @ W)

    def M6(k):

        A = sphere.operator('Id-Cos')
        B = sphere.operator('Id+Cos')

        p = k // (abs(k) + (k==0))

        W = (A(p) @ B(-p))**abs(k)

        I = sphere.operator('Id')
        Z = sphere.operator('Cos')

        phi = ( 1 + 5**(1/2) ) / 2

        if abs(k) == 0:
            P = (1/16) * (-5*I + 105*Z**2 - 315*Z**4 + 231*Z**6)
            P *= 13
            P *= (851741 - 665280 * phi) / 20720464

        if abs(k) == 1:
            P = (15/16) * (I - 18*Z**2 + 33*Z**4)
            P *= 91/60
            P *= (-6546903 + 3114606 * phi) / 20720464

        if abs(k) == 2:
            P = (3/2) * (-I + 11*Z**2)
            P *= 91/128
            P *= (-851741 + 665280 * phi) / 2590058

        if abs(k) == 3:
            P = I
            P *= 3003/1024
            P *= (198391 - 94382 * phi) / 1295029

        return P @ W

    def right_blocks(Lmax,n,m):

        if (abs(n-m) > 6) or ((n-m) % 2 == 1):
            return infinite_csr(np.zeros((Lmax+1-abs(n),Lmax+1-abs(m))))

        Right = M6((n-m)//2)

        if n == m:
            Right += sphere.operator('Id')

        return Right(Lmax,m,0)[:-6]

    def left_blocks(Lmax,n,m):

        if (abs(n-m) > 12) or ((n-m) % 2 == 1):
            return infinite_csr(np.zeros((Lmax+1-abs(n),Lmax+1-abs(m))))

        Left = 0

        for p in range(-Lmax,Lmax+1):

            k = (n-p)
            j = (p-m)

            if (abs(k) <= 6) and (k % 2 == 0) and (abs(j) <= 6) and (j % 2 == 0) :

                k //= 2
                j //= 2

                Left += D(-1) @ M6(k) @ S2(+1,+j) @ D(-1)
                Left += D(+1) @ M6(k) @ S2(-1,-j) @ D(+1)

        if abs(n - m) <= 6:

            k = (n-m)//2

            Left += D(-1) @ M6(k) @ D(+1) + D(+1) @ M6(k) @ D(-1)

            Left += D(-1) @ S2(+1,k) @ D(-1) + D(+1) @ S2(-1,-k) @ D(+1)

        if n == m:

            Left +=  D(-1) @ D(+1) + D(+1) @ D(-1)

        return Left(Lmax,m,0)[:-12]/2

    def matrices(Lmax):
        Right, Left = [], []
        for n in range(-Lmax,Lmax+1):
            Right += [[right_blocks(Lmax,n,m).A.astype(np.float64) for m in range(-Lmax,Lmax+1)]]
            Left  += [[left_blocks(Lmax,n,m).A.astype(np.float64)  for m in range(-Lmax,Lmax+1)]]
        return np.bmat(Left), np.bmat(Right)


    Lmax = 30

    LHS, RHS = matrices(Lmax)
    PDE_eigs = np.sort(np.abs(sp.linalg.eigh(LHS,b=RHS)[0]))[:60]

elif calculate_data == "False":
    PDE_eigs = np.array([0.      ,  0.991754,  0.991754,  0.991754,  2.954781,  2.954781,
           2.954781,  2.954781,  2.954781,  5.195832,  5.195832,  5.195832,
           6.697312,  6.697312,  6.697312,  6.697312,  9.794575,  9.794575,
           9.794575,  9.794575,  9.794575, 10.563822, 10.563822, 10.563822,
          10.563822, 14.790965, 14.790965, 14.790965, 15.061644, 15.061644,
          15.061644, 15.061644, 15.061644, 15.567727, 15.567727, 15.567727,
          20.700274, 20.700274, 20.700274, 21.195208, 21.195208, 21.195208,
          21.195208, 21.248118, 21.248118, 21.248118, 21.248118, 21.248118,
          21.419419, 27.932443, 27.932443, 27.932443, 27.932443, 27.932443,
          28.004496, 28.004496, 28.004496, 28.004496, 28.249831, 28.249831])

plt.rc('axes', linewidth=3)

fontsize=20
legend_ncol=1
markersize=18
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig = plt.figure(figsize=(13, 10))
ax = fig.add_subplot(111)

end = np.sum(2 * np.arange(1, 4) + 1) + 1

sph_harm_eigs = np.array([0,
                          1., 1., 1.,
                          1.73205081, 1.73205081, 1.73205081, 1.73205081, 1.73205081,
                          2.44948974, 2.44948974, 2.44948974, 2.44948974, 2.44948974, 2.44948974, 2.44948974])**2

ODE_eigs = np.array([0,
                     0.9981824, 0.9981824, 0.9981824,
                     1.71217321, 1.71217321, 1.71217321, 1.71217321, 1.71217321,
                     2.26124878, 2.26124878, 2.26124878, 2.51844415, 2.51844415, 2.51844415, 2.51844415])**2

ax.plot(ODE_eigs, 'o', markersize=markersize, label="$k_{jm}^{2}$",
        c=colors[0])

ax.plot(sph_harm_eigs, 'o', markersize=markersize, label="$\\lambda^{(0)}_{j}$",
        c=colors[1], alpha = 0.6)

ax.plot(PDE_eigs[:end], 'o', markersize=markersize, label="$\\tilde{k}_{jm}$",
        c=colors[1])

tidy_visual_comparison(ax)

ax.set_ylim([-0.3, 6.95])
plt.yticks(fontsize=26, weight='bold')
ax.set_xticklabels([])

plt.savefig("soccer_ball_eigs.png", bbox_inches='tight', dpi = 150)
