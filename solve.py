from graph_modules.modules import *
import sys

gtype = sys.argv[1]

if gtype == "spiderweb":
    g = SpiderWeb(2 * 100, 100, j=0)

elif gtype == "soccer_ball":
    g = Icosphere(("Truncate", 1), plot=False)

guesses = np.linspace(1, 3, 3)

eigs = Newton_Runner(g, guesses, printerval=50, max_iters=500, atol=10)