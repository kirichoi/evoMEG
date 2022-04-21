# -*- coding: utf-8 -*-
"""
CCR analysis extension module

Kiri Choi (c) 2018
"""

import tellurium as te
import roadrunner
import numpy as np
from scipy import signal
from sklearn import neighbors
import plotting as pt
import networkx as nx


def getWeights(dist):
    """
    """



def ensembleFluxControlCoefficient(model_col):
    """
    """

    r = te.loada(model_col[0])
    
    F = np.empty((len(model_col), r.getNumReactions(), r.getNumReactions()))
    
    for i in range(len(model_col)):
        r = te.loada(model_col[i])
        r.steadyState()
        F[i] = r.getScaledFluxControlCoefficientMatrix()
    
    return F
    

def ensemblePredictionMetric(realModel, model_col, predictionMetric=['SS', 'R', 'T', 'F']):
    """
    """
    
    realSS, realR, realT, realF = testModelAnalysis(realModel)
    
    if 'SS' in predictionMetric:
        SS = ensembleSteadyState(model_col)
    elif 'R' in predictionMetric:
        R = ensembleReactionRates(model_col)
    elif 'T' in predictionMetric:
        T = ensembleTimeCourse(model_col)
    elif 'F' in predictionMetric:
        F = ensembleFluxControlCoefficient(model_col)        

    
def ensembleReactionRates(model_col):
    """
    """
    
    r = te.loada(model_col[0])
    
    J = np.empty((len(model_col), r.getNumReactions()))
    
    for i in range(len(model_col)):
        r = te.loada(model_col[i])
        r.steadyState()
        J[i] = r.getReactionRates()
    
    return J


def ensembleSteadyState(model_col):
    """
    """
    
    r = te.loada(model_col[0])
    
    SS = np.empty((len(model_col), r.getNumFloatingSpecies()))
    
    for i in range(len(model_col)):
        r = te.loada(model_col[i])
        r.steadyState()
        SS[i] = r.getFloatingSpeciesConcentrations()
    
    return SS


def ensembleTimeCourse(model_col):
    """
    """
    
    r = te.loada(model_col[0])
    
    T = np.empty((len(model_col), 100, r.getNumFloatingSpecies()))
    
    for i in range(len(model_col)):
        r = te.loada(model_col[i])
        T[i] = r.simulate(0, 100, 100)[:,1:]
    
    return T


def isConnected(rl):
    """
    Check if a reaction list features connected graph.
    
    :param rl: reaction list
    """
    
    G = nx.Graph()
    
    for i in range(len(rl)):
        for j in range(len(rl[i][3])):
            G.add_edges_from([(rl[i][3][j], str(i))])
        
        for k in range(len(rl[i][4])):
            G.add_edges_from([(str(i), rl[i][4][k])])
        
        for l in range(len(rl[i][5])):
            G.add_edges_from([(rl[i][5][l], str(i))])
            
        for m in range(len(rl[i][6])):
            G.add_edges_from([(rl[i][6][m], str(i))])
    
    return nx.is_connected(G)


def selectWithCutoff(model_top, dist_top, cutoff=0.1):
    """
    Model selection routine that returns a list of models with distances within
    the defined percentage.
    
    :param model_top: list of models sorted according to corresponding distances
    :param dist_top: list of sorted distances
    :param cutoff: percentage to cutoff
    """
    
    coind = int(len(model_top)*cutoff)
    pt.plot_distance_histogram(dist_top, cutoff_val=dist_top[coind])
    
    return model_top[:coind], dist_top[:coind]


def selectWithKernalDensity(model_top, dist_top, export_flag=False):
    """
    Model selection rountine that returns a list of models based on the output
    of kernal density estimation.
    
    :param model_top: list of models sorted according to corresponding distances
    :param dist_top: list of sorted distances
    """
    

    dist_top_reshape = dist_top.reshape((len(dist_top),1))
    
    kde_xarr = np.linspace(0, np.max(dist_top), int(np.max(dist_top)*10))[:, np.newaxis]
    
    kde = neighbors.KernelDensity(kernel='gaussian', bandwidth=0.3).fit(dist_top_reshape)
    
    log_dens = kde.score_samples(kde_xarr)
    
    minInd = signal.argrelextrema(log_dens, np.less)
    
    if len(minInd[0]) == 0:
        minInd = np.array([[len(model_top) - 1]])

    if export_flag:
        minInd = np.array([[len(model_top) - 1]])
    
    kde_idx = (np.abs(dist_top - kde_xarr[minInd[0][0]])).argmin()
    
    return minInd, log_dens, kde_xarr, kde_idx


    
def testModelAnalysis(realModel):
    """
    """
    



