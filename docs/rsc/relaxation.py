import numpy as np
import matplotlib.pyplot as plt


START_Y = 0.0
TARGET_Y = 1.0
C_0 = 4.0
C_1 = 0.0

def relax_constant(i, n):
    return C_0 / n


def relax_tanh(i, n):
    x = (i + 1) / n
    return relax_constant(i, n) + C_1 * (1 / n) * np.tanh(x)


def relax(y0, y1, i, n, backend=relax_constant):
    f = backend(i, n)
    return (1. - f) * y0 + f * y1


def oraculus(iters, backend=relax_constant):
    x = np.arange(0, iters + 1)
    y = [START_Y, ]
    f = [0.0, ]
    for i in range(iters - 1):
        y.append(relax(y[-1], TARGET_Y, i, iters, backend=backend))
        f.append(backend(i, iters))
    y.append(TARGET_Y)
    f.append(1.0)
    return x, np.asarray(y), np.asarray(f)


if __name__ == '__main__':
    for backend in (relax_tanh, ):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        for n, c in zip((10, 100, 1000), ('k', 'r', 'b')):
            x, y, f = oraculus(n, backend=backend)
            ax1.plot(x / n, y, color=c, label=f'M = {n}, C_0 = {C_0}')
            ax2.plot(x[1:-1] / n, 1 - f[1:-1], color=c,
                     label=f'M = {n}, C_0 = {C_0}')

        ax1.set_ylabel(r'$\mathrm{d} r / \mathrm{d} t$')
        ax1.set_xlim(0.0, 1.0)
        ax1.set_ylim(0.0, 1.0)
        ax2.set_xlabel(r'$m / M$')
        ax2.set_ylabel('$1 - f$')
        ax2.set_xlim(0.0, 1.0)
        # ax2.set_ylim(0.0, 1.0)
        ax1.grid()
        ax2.grid()
        ax1.legend(loc='best')
        plt.show()
