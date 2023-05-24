# laplace_operator_metric_graph

This codebase contains python3 scripts to produce numerical results in "A continuum limit of the Laplace operator on metric graphs" (Main Text).

Required modules are 2022-onwards versions of numpy, scipy, matplotlib, pickle.

To produce spiderweb figures 2a-b, run "python3 spiderweb.py". This loads eigenmode data from ``efs_ODE.pkl, eigs_ODE.pkl, eigs_PDE.pkl". 

This data is calculated by running Newton's method on $det(L(k)) = 0$ based off eq.4 in the Main Text $L(k)f(V) = 0$--quite an involved process.

An example of this in action is given by running "python3 solve.py arg", where arg can be either "spiderweb" or "soccer_ball". This will produce values of $k$ starting from np.linspace(1, 3, 3).

The soccer ball script produces figure2g. It requires the sphere_m_ops branch of Dedalus3: https://dedalus-project.readthedocs.io/en/latest/.
To run, run "python3 soccer_ball.py arg" where arg can be either "True" or "False", depending on whether or not you want to solve the PDE (Supplemental Material [81]), or to import the already calculated data.
