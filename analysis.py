# -*- coding: utf-8 -*-
"""
CCR analysis extension module

Kiri Choi (c) 2018
"""

import numpy as np
import tellurium as te
import networkGenerator as ng


def getWeights(dist):
    """
    """

def customGetScaledConcentrationControlCoefficientMatrix(r):
    """
    Numpy implementation of GetScaledConcentrationControlCoefficientMatrix()
    that does not force run steadyState()
    
    Parameters
    ----------
    r : roadrunner.RoadRunner()
        roadrunner instance.

    Returns
    -------
    T5 : numpy.ndarray
        Scaled concentration control coefficient matrix.
    """
    
    re = r.simulate(0, 10000, 5)
    # r.steadyState()
    uelast = r.getUnscaledElasticityMatrix()
    Nr = r.getNrMatrix()
    T1 = np.dot(Nr, uelast)
    LinkMatrix = r.getLinkMatrix()
    Jac = np.dot(T1, LinkMatrix)
    T2 = np.negative(Jac)
    Inv = np.linalg.inv(T2)
    T3 = np.dot(Inv, Nr)
    T4 = np.dot(LinkMatrix, T3)

    a = np.tile(r.getReactionRates(), (np.shape(T4)[0], 1))
    b = np.tile(r.getFloatingSpeciesConcentrations(), (np.shape(T4)[1], 1)).T
    
    T5 = np.multiply(T4, np.divide(a, b))
    
    return T5


def cacheFallBack1(Settings, ens_model):
    
    ens_dist = np.empty(Settings.ens_size)
    ens_concCC = np.empty(Settings.ens_size, dtype='object')
    
    for i in range(len(ens_model)):
        r = te.loada(ens_model[i])
        concCC = customGetScaledConcentrationControlCoefficientMatrix(r)
        
        dist_obj = ((np.linalg.norm(Settings.realConcCC - concCC))*
                    (1 + np.sum(np.not_equal(np.sign(Settings.realConcCC), np.sign(concCC)))))
        ens_dist[i] = dist_obj
        ens_concCC[i] = concCC
        
    
    return (ens_dist, ens_concCC)


def cacheFallBack2(Settings, ens_model, ns, nr):
    
    ens_stoi = np.empty((Settings.ens_size, ns, nr), dtype=int)
    ens_rtypes = np.empty((Settings.ens_size, 3, nr), dtype=int)
    ens_ia = np.empty((Settings.ens_size, ns, nr), dtype=int)
    tracking = []
    
    for i in range(len(ens_model)):
        (stoi, rStoi, rtypes, ia) = ng.generateSTFromAntimony(ens_model[i])
        
        ens_stoi[i] = stoi
        ens_rtypes[i] = rtypes
        ens_ia[i] = ia
        tracking.append(rStoi.tolist())
        
    
    return (ens_stoi, ens_rtypes, ens_ia, tracking)


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
    Check if a reaction list is equivalent to a connected graph.
    
    :param rl: reaction list
    """
    
    import networkx as nx
    
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
    
    import plotting as pt
    
    coind = int(len(model_top)*cutoff)
    pt.plot_distance_histogram(dist_top, cutoff_val=dist_top[coind])
    
    return model_top[:coind], dist_top[:coind]


def selectWithKernalDensity(dist_top):
    """
    Model selection rountine that returns a list of models based on the output
    of kernal density estimation.
    
    :param model_top: list of models sorted according to corresponding distances
    :param dist_top: list of sorted distances
    """
    
    from scipy import signal
    from sklearn import neighbors
    
    x = np.linspace(0, np.max(dist_top), int(np.max(dist_top)*10))

    dist_top_reshape = dist_top.reshape((len(dist_top),1))
    
    kde_xarr = x[:, np.newaxis]
    
    kde = neighbors.KernelDensity(kernel='gaussian', bandwidth=0.3).fit(dist_top_reshape)
    
    log_dens = kde.score_samples(kde_xarr)
    
    extremaInd = signal.argrelextrema(log_dens, np.less)[0]
    
    if len(extremaInd) == 0:
        extremaInd = [len(x)-1]
    
    dist = dist_top - x[extremaInd[0]]
    
    minInd = np.where(dist <= 0)[0][-1]
    
    return [minInd, log_dens, kde_xarr.flatten(), extremaInd[0]]


    
def testModelAnalysis(realModel):
    """
    """
    



