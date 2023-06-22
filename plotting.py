# -*- coding: utf-8 -*-
"""
CCR plotting module

Kiri Choi (c) 2018
"""

import os
import tellurium as te
import numpy as np
import matplotlib.pyplot as plt


def plotAllProgress(listOfDistances, labels=None, show=True, SAVE_PATH=None):
    """
    Plots multiple distance progressions 
    
    :param listOfDistances: 2D array of recorded distances
    :param labels: list of strings to use as labels
    :param SAVE_PATH: path to save the plot
    """
    
    for i in range(len(listOfDistances)):
        plt.plot(listOfDistances[i])
    if labels:
        plt.legend(labels)
    plt.xlabel("Generations", fontsize=15)
    plt.ylabel("Distance", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if SAVE_PATH is not None:
        if os.path.splitext(SAVE_PATH)[1] == '':
            plt.savefig(os.path.join(SAVE_PATH, 'images/convergence.pdf'), bbox_inches='tight')
        else:
            plt.savefig(SAVE_PATH, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
        

def plotProgress(distance, show=True, SAVE_PATH=None):
    """
    Plots a distance progression
    
    :param distance: array of recorded distances
    :param model_type: reference model type, e.g. 'FFL', 'Linear', etc.
    :param SAVE_PATH: path to save the plot
    """
    
    plt.plot(distance)
    plt.xlabel("Generations", fontsize=15)
    plt.ylabel("Distance", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if SAVE_PATH is not None:
        if os.path.splitext(SAVE_PATH)[1] == '':
            plt.savefig(os.path.join(SAVE_PATH, 'images/convergence.pdf'), bbox_inches='tight')
        else:
            plt.savefig(SAVE_PATH, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
        

def plotResidual(realModel, ens_model, ens_dist, show=True, SAVE_PATH=None):
    """
    Plots residuals
    
    :param realModel: reference model
    :param ens_model: model ensemble
    :param ens_dist: model distances
    :param model_typ: reference model type
    :param SAVE_PATH: path to save the plot
    """
    
    r_real = te.loada(realModel)
    result_real = r_real.simulate(0, 100, 100)
    
    top_result = []
    top_diff = []
    
    for i in range(len(ens_model)):
        r = te.loada(ens_model[np.argsort(ens_dist)[i]])
        top_sim = r.simulate(0, 100, 100)
        top_result.append(top_sim)
        top_diff.append(np.subtract(result_real[:,1:], top_sim[:,1:]))

    percentage = 0.1#float(pass_size)/ens_size
    
    ave_diff = np.average(top_diff[:int(len(ens_model)*percentage)], axis=0)
    
    plt.plot(ave_diff)
    plt.xlabel("Time (s)", fontsize=15)
    plt.ylabel("Residual", fontsize=15)
    plt.legend(r.getFloatingSpeciesIds())
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if SAVE_PATH is not None:
        if os.path.splitext(SAVE_PATH)[1] == '':
            plt.savefig(os.path.join(SAVE_PATH, 'images/average_residual.pdf'), bbox_inches='tight')
        else:
            plt.savefig(SAVE_PATH, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
        
    
def plotDistanceHistogram(ens_dist, nbin=25, show=True, SAVE_PATH=None):
    """
    """
    
    plt.hist(ens_dist, bins=nbin, density=True)
    plt.xlabel("Distance", fontsize=15)
    plt.ylabel("Normalized Frequency", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if SAVE_PATH is not None:
        if os.path.splitext(SAVE_PATH)[1] == '':
            plt.savefig(os.path.join(SAVE_PATH, 'images/distance_hist.pdf'), bbox_inches='tight')
        else:
            plt.savefig(SAVE_PATH, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
        

def plotDistanceHistogramWithKDE(kdeOutput, dist_top, nbin=40, show=True, SAVE_PATH=None):
    """
    """
    
    x = np.linspace(0, np.max(dist_top), int(np.max(dist_top)*10))
    hist = plt.hist(dist_top, bins=nbin, density=True)
    plt.vlines(x[kdeOutput[0][0]], 0, np.max(hist[0]), 
               linestyles='dashed', color='tab:green')
    plt.plot(kdeOutput[2], np.exp(kdeOutput[1]), color='tab:red')
    plt.xlabel("Distance", fontsize=15)
    plt.ylabel("Normalized Frequency", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if SAVE_PATH is not None:
        if os.path.splitext(SAVE_PATH)[1] == '':
            plt.savefig(os.path.join(SAVE_PATH, 'images/distance_hist_w_KDE.pdf'), bbox_inches='tight')
        else:
            plt.savefig(SAVE_PATH, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
        

def plotNetwork(path, scale=1.5):
    """
    Plot a network diagram
    
    :param path: path to a model
    :param scale: diagram scale
    """
    
    import netplotlib as npl
    
    net = npl.Network(path)
    net.scale = scale
    net.draw()


def plotMemoryUsage(memory, show=True, SAVE_PATH=None):
    """
    Plot memory usage

    :param memory: list of used memory
    :param SAVE_PATH: path to save the plot
    """
    
    fig = plt.figure(figsize=(6,4))
    plt.plot(np.divide(np.subtract(memory, memory[0]), 1e6)[1:])
    plt.xlabel("Generation", fontsize=15)
    plt.ylabel("Approx. Memory usage (MB)", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if SAVE_PATH is not None:
        if os.path.splitext(SAVE_PATH)[1] == '':
            plt.savefig(os.path.join(SAVE_PATH, 'images/memoryUsage.pdf'), bbox_inches='tight')
        else:
            plt.savefig(SAVE_PATH, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
        

def plotNetworkEnsemble(path, index=None, threshold=0., scale=1.5):
    """
    Plot network ensemble diagram
    
    :param path: path to output folder
    :param index: index of models to be included in the diagram
    :param threshold: threshold of reactions to be plotted
    :param scale: diagram scale
    """
    
    import netplotlib as npl
    
    model_dir = os.path.join(path, 'models')
    
    modelfiles = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]
    
    modelcontent = []
    for i in modelfiles:
        sbmlstr = open(os.path.join(model_dir, i), 'r')
        modelcontent.append(sbmlstr.read())
        sbmlstr.close()
    
    if index >= len(modelcontent):
        raise Exception("Specified index value is larger than the size of the list")
    
    net = npl.NetworkEnsemble(modelcontent[:index])
    net.plottingThreshold = threshold
    net.scale = scale
    net.drawWeightedDiagram()


def plotConcCC(model, show=True, SAVE_PATH=None):
    
    try:
        r = te.loada(model)
    except:
        try:
            r = te.loads(model)
        except:
            raise Exception("Not a valid model")
    
    try:
        concCC = r.getScaledConcentrationControlCoefficientMatrix()
    except:
        raise Exception("Cannot calculate control coefficients")
                
    fig, ax = plt.subplots(figsize=(6,4))
    im = ax.imshow(concCC, interpolation='none', cmap='RdBu')
    ax.set_yticks(np.arange(np.shape(concCC)[0]), concCC.rownames)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(np.shape(concCC)[1]), concCC.colnames)
    
    for i in range(concCC.shape[0]):
        for j in range(concCC.shape[1]):
            text = ax.text(j, i, '{:.2f}'.format(concCC[i,j]), ha="center", va="center", color="k")
    
    if SAVE_PATH is not None:
        if os.path.splitext(SAVE_PATH)[1] == '':
            plt.savefig(os.path.join(SAVE_PATH, 'images/controlCoeff.pdf'), bbox_inches='tight')
        else:
            plt.savefig(SAVE_PATH, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    
    