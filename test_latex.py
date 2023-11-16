import matplotlib.pyplot as plt


# Enable LaTeX rendering in Matplotlib
plt.rc('text', usetex=True)
plt.plot([1, 2, 3], [4, 5, 6], label=r'$\alpha$-Beta')
plt.xlabel(r'$\theta$')
plt.title(r'Plot with $\LaTeX$ Labels')
plt.legend()
plt.show()
