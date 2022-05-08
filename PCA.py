#!/usr/bin/env python3
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
import time



## Direccion de la base de datos del sistema
path_cancer = 'ovariancancer_obs.csv'
delimitador = ','


path_diabetes = "diabetes-dataset.csv"
## Base datos con definicion de cual es cancer o no
f = open('ovariancancer_grp.csv', "r")
grp = f.read().split("\n")

def read_data(path, limitador):
    obs = np.loadtxt(path, delimiter=limitador)
    return obs

def fancy_plots():
    # Define parameters fancy plot
    pts_per_inch = 72.27
    # write "\the\textwidth" (or "\showthe\columnwidth" for a 2 collumn text)
    text_width_in_pts = 300.0
    # inside a figure environment in latex, the result will be on the
    # dvi/pdf next to the figure. See url above.
    text_width_in_inches = text_width_in_pts / pts_per_inch
    # make rectangles with a nice proportion
    golden_ratio = 0.618
    # figure.png or figure.eps will be intentionally larger, because it is prettier
    inverse_latex_scale = 2
    # when compiling latex code, use
    # \includegraphics[scale=(1/inverse_latex_scale)]{figure}
    # we want the figure to occupy 2/3 (for example) of the text width
    fig_proportion = (3.0 / 3.0)
    csize = inverse_latex_scale * fig_proportion * text_width_in_inches
    # always 1.0 on the first argument
    fig_size = (1.0 * csize, 0.7 * csize)
    # find out the fontsize of your latex text, and put it here
    text_size = inverse_latex_scale * 10
    label_size = inverse_latex_scale * 10
    tick_size = inverse_latex_scale * 8

    params = {'backend': 'ps',
            'axes.labelsize': text_size,
            'legend.fontsize': tick_size,
            'legend.handlelength': 2.5,
            'legend.borderaxespad': 0,
            'xtick.labelsize': tick_size,
            'ytick.labelsize': tick_size,
            'font.family': 'serif',
            'font.size': text_size,
            # Times, Palatino, New Century Schoolbook,
            # Bookman, Computer Modern Roman
            # 'font.serif': ['Times'],
            'ps.usedistiller': 'xpdf',
            'text.usetex': True,
            'figure.figsize': fig_size,
            # include here any neede package for latex
            'text.latex.preamble': [r'\usepackage{amsmath}',
                ],
                }
    plt.rcParams.update(params)
    plt.clf()
    # figsize accepts only inches.
    fig = plt.figure(1, figsize=fig_size)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.13,
                        hspace=0.21, wspace=0.02)
    plt.ioff()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    return fig, ax1, ax2

def fancy_plots_1():
    # Define parameters fancy plot
    pts_per_inch = 72.27
    # write "\the\textwidth" (or "\showthe\columnwidth" for a 2 collumn text)
    text_width_in_pts = 300.0
    # inside a figure environment in latex, the result will be on the
    # dvi/pdf next to the figure. See url above.
    text_width_in_inches = text_width_in_pts / pts_per_inch
    # make rectangles with a nice proportion
    golden_ratio = 0.618
    # figure.png or figure.eps will be intentionally larger, because it is prettier
    inverse_latex_scale = 2
    # when compiling latex code, use
    # \includegraphics[scale=(1/inverse_latex_scale)]{figure}
    # we want the figure to occupy 2/3 (for example) of the text width
    fig_proportion = (3.0 / 3.0)
    csize = inverse_latex_scale * fig_proportion * text_width_in_inches
    # always 1.0 on the first argument
    fig_size = (1.0 * csize, 0.7 * csize)
    # find out the fontsize of your latex text, and put it here
    text_size = inverse_latex_scale * 10
    label_size = inverse_latex_scale * 10
    tick_size = inverse_latex_scale * 8

    params = {'backend': 'ps',
            'axes.labelsize': text_size,
            'legend.fontsize': tick_size,
            'legend.handlelength': 2.5,
            'legend.borderaxespad': 0,
            'xtick.labelsize': tick_size,
            'ytick.labelsize': tick_size,
            'font.family': 'serif',
            'font.size': text_size,
            # Times, Palatino, New Century Schoolbook,
            # Bookman, Computer Modern Roman
            # 'font.serif': ['Times'],
            'ps.usedistiller': 'xpdf',
            'text.usetex': True,
            'figure.figsize': fig_size,
            # include here any neede package for latex
            'text.latex.preamble': [r'\usepackage{amsmath}',
                ],
                }
    plt.rcParams.update(params)
    plt.clf()
    # figsize accepts only inches.
    fig = plt.figure(1, figsize=fig_size)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.13,
                        hspace=0.05, wspace=0.02)
    plt.ioff()
    ax1 = fig.add_subplot(111)
    return fig, ax1

def media_vector(data):
    matrix = data
    ones_aux = np.ones((1,matrix.shape[0]))
    mean_aux = np.dot(ones_aux,matrix)/matrix.shape[0]
    ones_col = np.ones((matrix.shape[0],1))
    mean_matrix = np.dot(ones_col,mean_aux)
    aux = matrix-mean_matrix
    C = (1/(matrix.shape[0]-1))*aux.T@aux
    return C


def media_vector_1(data):
    matrix = data
    matrix = matrix.T
    ones_aux = np.ones((matrix.shape[1],1))
    mean_aux = np.dot(matrix,ones_aux)/matrix.shape[1]
    aux = matrix-mean_aux
    C = (1/(matrix.shape[1]-1))*aux@aux.T
    return C

def graficas_energia(S,S1):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(121)
    ax1.plot(np.cumsum(S)/np.sum(S),'-o',color='k')

    ax2 = fig1.add_subplot(122)
    ax2.plot(np.cumsum(S1)/np.sum(S1),'-o',color='k')
    plt.show()


def new_dimention(v, data):
    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')
    for j in range(data.shape[0]):
        x =  np.dot(data[j,:],v[:,0])
        y =  np.dot(data[j,:],v[:,1])
        z =  np.dot(data[j,:],v[:,2])
        if data[j,-1] == 1:
            ax.scatter(x,y,z,marker='x',color='r',s=50)
        else:
            ax.scatter(x,y,z,marker='o',color='b',s=50)
    ax.view_init(25,20)
    plt.show()

def fancy_plot_eig(S):
    fig1, ax1, ax2 = fancy_plots()
    ## Add values figure
    eig_values_plot, = ax1.semilogy(S,
                    color='#00429d', lw=2, ls="-")
    energy_plot, = ax2.plot(np.cumsum(S)/np.sum(S),
                    color='#9e4941', lw=2, ls="-.")
    
    ## Label of the first plot
    ax1.set_ylabel(r"$\textrm{Singular Values}$", rotation='vertical')
    ax1.legend([eig_values_plot],
            [r'$\sigma_r$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax1.grid(color='#949494', linestyle='-.', linewidth=0.5)

    ## Label of the second plot
    ax2.set_ylabel(r"$\textrm{Acumulaive}$", rotation='vertical')
    ax2.set_xlabel(r'$[r]$', labelpad=5)
    ax2.legend([energy_plot],
            [r'$energy$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax2.grid(color='#949494', linestyle='-.', linewidth=0.5)

    fig1.savefig("energy_and_eig_values.eps", resolution=300)
    fig1.savefig("energy_and_eig_values.png", resolution=300)
    fig1
    plt.show()


def fancy_plots_matrix(covariance_matrix):
    fig2, ax11 = fancy_plots_1()
    ## Axis definition necesary to fancy plots
    ax11.matshow(covariance_matrix)

    ax11.set_ylabel(r"$\textrm{Features}$", rotation='vertical')
    ax11.set_xlabel(r"$\textrm{Features}$", labelpad=5)

    fig2.savefig("covariance_matrix.eps", resolution=300)
    fig2.savefig("covariance_matrix.png", resolution=300)
    fig2
    plt.show()

def normilize_matrix(x):
    xmax, xmin = x.max(), x.min()
    x = (x - xmin)/(xmax - xmin)
    return x

def main():
    ## Load Data 
    data_cancer = read_data(path_cancer, delimitador)

    ## Covariance Matrix
    C = media_vector(data_cancer)
    C_1 = media_vector_1(data_cancer)
    C_3 = np.cov(data_cancer.T)

    ## eig values and vectors
    tic = time.time()
    e_1, v_1 = LA.eig(C_3)
    toc = tic-time.time()
    print(toc)
    e_1 = np.absolute(e_1)
    v_1 = np.absolute(v_1)

    ## Verificar norm = 1
    u_1 = v_1[:,0]
    u_2 = v_1[:,1]

    norm_1 = np.dot(u_1,u_1)
    norm_2 = np.dot(u_2,u_2)

    ## Fancy plots
    fancy_plot_eig(e_1)

    ## fancy plot matrix
    fancy_plots_matrix(C)
    fancy_plots_matrix(C_3)


    ## Figure new dimention
    new_dimention(v_1, data_cancer)
    


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Press ctrl-c to end the statement")

