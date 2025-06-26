from graph_modules.modules import *

eigs_PDE = pickle_load("eigs_PDE")
eigs_ODE = pickle_load("eigs_ODE")
efs_ODE = pickle_load("efs_ODE")

num_Vs_per_dim = np.int64(10**np.linspace(1.2, 4, 18))[2:-8]

print("\n", 64*"-")
print("(j x m)=(3 x 3) GRID OF NORM OF L(k) @ f(V) FOR EACH GRAPH SIZE:")
print(64*"-", "\n")

for eig_num_V in num_Vs_per_dim[:5]:

    results = np.zeros((3, 3))
    print(f"Number of radial vertices = {eig_num_V}\n")

    g = SpiderWeb(eig_num_V * 2, eig_num_V)

    for j in range(3):
        g.j = j

        for m in range(3):
            results[j, m] = np.linalg.norm(g.construct_L(eigs_ODE[eig_num_V][j, m]) @ efs_ODE[eig_num_V][j][m])

    print(results, "\n")

eigs_rel_err = np.zeros((3, 3, len(num_Vs_per_dim)))
efs_rel_err = np.zeros((3, 3, len(num_Vs_per_dim)))

for en, num_vs_per_dim in enumerate(num_Vs_per_dim):

    g = SpiderWeb(num_vs_per_dim * 2, num_vs_per_dim)

    for j in range(3):

        for m in range(3):

            eigs_rel_err[j, m, en] = rel_err(eigs_PDE[j, m], eigs_ODE[num_vs_per_dim][j, m])

            ODE_f = efs_ODE[num_vs_per_dim][j][m].flatten()
            PDE_f = calculate_radial_EF_pde(g.r, j, m, eigs_PDE).flatten()

            dif0 = np.linalg.norm(ODE_f - PDE_f)
            dif1 = np.linalg.norm(ODE_f + PDE_f)
            dif = np.min((dif0, dif1))

            efs_rel_err[j, m, en] = dif

index_for_colors = 0
linewidth = 6
fontsize = 18
labelsize = 16
markersize = 20

plot0 = Plot_Loglog(figsize=(9, 11))
colors_for_table = np.array(plot0.colors[:-1]).reshape(3,3).T.flatten()
plt.rc('axes', linewidth=3)

for j in range(3):
    for m in range(3):
        plot0(num_Vs_per_dim * 2, eigs_rel_err[j, m, :], label=f"mn = {j}{m}",
              c=colors_for_table[index_for_colors], markersize=markersize, linewidth=2.5)
        index_for_colors += 1

plot0.ax.tick_params(axis='both', which='major', labelsize=labelsize)
plot0.ax.tick_params(axis='both', which='minor', labelsize=labelsize)

plt.xticks(fontsize=22, weight='bold')
plt.yticks(fontsize=22, weight='bold')
plot0.ax.set_xticks([], minor=True)
plot0.ax.set_yticks([], minor=True)

ax1 = plot0.ax.twiny()
ax1.loglog(num_Vs_per_dim * 2, (num_Vs_per_dim * 2) ** (-1.) * 3,
           c='r', linewidth=linewidth, label="$|V|^{-1}$")
ax1.get_yaxis().set_visible(False)
ax1.set_xticks([])

# tablelegend(plot0.ax, ncol=3, bbox_to_anchor=(1, 1), fontsize=fontsize + 4,
#             row_labels=['$m=0$', '$m=1$', '$m=2$'],
#             col_labels=['$j=0$', '$j=1$', '$j=2$'])

plot0.ax.minorticks_off()
plot0.ax.tick_params('x', length=7, width=3, which='major')
plot0.ax.tick_params('y', length=7, width=3, which='major')

plot0.ax.set_ylim([3e-5, 3e-1])

plt.savefig("spider_web_eigs_err.png", bbox_inches='tight', dpi = 150)

index_for_colors = 0
plot0 = Plot_Loglog(figsize=(9, 11))
plt.rc('axes', linewidth=3)

for m in range(3):
    for j in range(3):
        if j + m != 4:
            plot0(num_Vs_per_dim * 2, efs_rel_err[j, m, :], label=f"mn = {j}{m}",
                  c=colors_for_table[index_for_colors], markersize=markersize, linewidth=2.5)
            index_for_colors += 1

plot0.ax.tick_params(axis='both', which='major', labelsize=labelsize)
plot0.ax.tick_params(axis='both', which='minor', labelsize=labelsize)

plt.xticks(fontsize=22, weight='bold')
plt.yticks(fontsize=22, weight='bold')
plot0.ax.set_yticks([], minor=True)

ax1 = plot0.ax.twiny()
ax1.loglog(num_Vs_per_dim * 2, (num_Vs_per_dim * 2) ** (-1.) * 10,
           c='r', linewidth=linewidth, label="$N_{r}^{-1}$")
ax1.loglog(num_Vs_per_dim * 2, (num_Vs_per_dim * 2) ** (-2.) / 3,
           c='b', linewidth=linewidth, label="$N_{r}^{-2}$")
ax1.get_yaxis().set_visible(False)
ax1.legend(loc=3, fontsize=fontsize + 7)
ax1.set_xticks([])
ax1.set_xticks([])

# plot0.ax.minorticks_off()
plot0.ax.tick_params('x', length=6, width=3, which='minor')
plot0.ax.tick_params('x', length=9, width=3, which='major')
plot0.ax.tick_params('y', length=9, width=3, which='major')

plot0.ax.set_ylim([2e-7, 2.5e-1])

plt.savefig("spider_web_efs_err.png", bbox_inches='tight', dpi = 150)