# -*- coding: utf-8 -*-
"""
Created on Sat May 19 16:01:19 2018

@author: kirichoi
"""

import os, sys, psutil, getopt
import tellurium as te
import roadrunner
import numpy as np
import antimony
import scipy.optimize
import networkGenerator as ng
import plotting as pt
import ioutils
import analysis
import time
import copy

#np.seterr(all='raise')

class SettingsClass:
    
    def __init__(self):
        # Input ===============================================================
        # Load experimental input or preconfigured settings
        
        # Path to a custom model (default: None)
        self.MODEL_INPUT = None
        # Path to experimental data - not implemented (default: None)
        self.DATA_INPUT = None
        # Path to preconfigured settings (default: None)
        self.READ_SETTINGS = None
        
        # Test models =========================================================
        # A selection of test models including (reversible/irreverisble) 
        # feedforward loop, linear chain, nested cycles, feedback loop, and
        # branched pathway
            
        # 'FFL_m', 'Linear_m', 'Nested_m', 'Branched_m', 'Feedback_m', 'sigPath'
        # 'FFL_r', 'Linear_r', 'Nested_r', 'Branched_r', 'Feedback_r'
        self.modelType = 'FFL_m'
        
        
        # General settings ====================================================
        # Settings for the population and the algorithm
        
        # Size of output ensemble
        self.ens_size = 50
        # Number of models used for recombination (default: int(ens_size/10))
        self.pass_size = int(self.ens_size/10)
        # Top percentage of population to track (default: 0.05)
        self.top_p = 0.05
        # Maximum iteration allowed for random generation (default: 200)
        self.maxIter_gen = 200
        # Maximum iteration allowed for mutation (default: 200)
        self.maxIter_mut = 200
        # Recombination probability (default: 0.3)
        self.recomb = 0.3
        # Track stoichiometry instead of reaction lists
        self.trackStoichiometry = True
        # Set conserved moiety (default: False)
        self.conservedMoiety = False
        
        
        # Termination criterion settings ======================================
        # Settings to control termination criterion
        
        # Maximum number of generations
        self.n_gen = 50
        # Number of generations w/o improvement
        self.gen_static = None
        # Threshold average distance
        self.thres_avg = None
        # Threshold median distance
        self.thres_median = None
        # Threshold shortest distance
        self.thres_shortest = None
        # Threshold top p-percent smallest distance
        self.thres_top = None
        
        # Optimizer settings ==================================================
        # Settings specific to differential evolution
        
        # Maximum iteration allowed (default: 1000)
        self.optiMaxIter = 1000
        # Optimizer tolerance (default: 1)
        self.optiTol = 1.
        # Run additional optimization at the end for polishing (default: False)
        self.optiPolish = False
        # Weight for control coefficients when calculating the distance - unused
        self.w1 = 16
        # Weight for steady-state and flux when calculating the distance - unused
        self.w2 = 1.0
        
        
        # Randomization settings ==============================================
        
        # Random seed
        self.r_seed = 123123
        # Flag to add Gaussian noise to the input
        self.NOISE = False
        # Standard deviation of absolute noise
        self.ABS_NOISE_STD = 0.005
        # Standard deviation of relative noise
        self.REL_NOISE_STD = 0.2
        
        
        # Plotting settings ===================================================
        
        # Flag to plot
        self.SHOW_PLOT = True
        # Flag to save plot
        self.SAVE_PLOT = True
        
        
        # Data settings =======================================================
            
        # Flag to collect all models in the ensemble
        self.EXPORT_ALL_MODELS = True
        # Flag to save collected models
        self.EXPORT_OUTPUT = True
        # Flag to save current settings
        self.EXPORT_SETTINGS = True
        # Path to save the output
        self.EXPORT_PATH = './outputs/stoi'
        # Overwrite the contents if the folder exists
        self.EXPORT_OVERWRITE = False
        # Create folders based on model names
        self.EXPORT_FORCE_MODELNAMES = False
        
        
        # Flag to run the algorithm - temporary
        self.RUN = False


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
    T4 : numpy.ndarray
        Scale concentration control coefficient matrix.
    """
    
    re = r.simulate(0, 10000, 5)
    r.steadyState()
    uelast = r.getUnscaledElasticityMatrix()
    Nr = r.getNrMatrix()
    T1 = np.dot(Nr, uelast)
    LinkMatrix = r.getLinkMatrix()
    Jac = np.dot(T1, LinkMatrix)
    T2 = Jac * (-1.0)
    Inv = np.linalg.inv(T2)
    T3 = np.dot(Inv, Nr)
    T4 = np.dot(LinkMatrix, T3)

    a = np.tile(r.getReactionRates(), (np.shape(T4)[0], 1))
    b = np.tile(r.getFloatingSpeciesConcentrations(), (np.shape(T4)[1], 1)).T
    
    T4 = np.multiply(T4, np.divide(a, b))
    
    return T4


def check_duplicate_reaction(data):
    ind = np.lexsort(data)
    diff = np.any(data.T[ind[1:]] != data.T[ind[:-1]], axis=1)
    edges = np.where(diff)[0] + 1
    result = np.split(ind, edges)
    result = [group for group in result if len(group) >= 2]
    return result


def f1(k_list, *args):
    global counts
    global countf
    
    args[0].reset()
    args[0].setValues(args[0].getGlobalParameterIds(), k_list)
    
    try:
        re = args[0].simulate(0, 10000, 5)
        uelast = args[0].getUnscaledElasticityMatrix()
        Nr = args[0].getNrMatrix()
        T1 = np.dot(Nr, uelast)
        LinkMatrix = args[0].getLinkMatrix()
        Jac = np.dot(T1, LinkMatrix)
        T2 = Jac * (-1.0)
        Inv = np.linalg.inv(T2)
        T3 = np.dot(Inv, Nr)
        T4 = np.dot(LinkMatrix, T3)
        
        a = np.tile(args[0].getReactionRates(), (np.shape(T4)[0], 1))
        b = np.tile(args[0].getFloatingSpeciesConcentrations(), (np.shape(T4)[1], 1)).T
        
        objCCC = np.multiply(T4, np.divide(a, b))
        
        if np.isnan(objCCC).any():
            dist_obj = 1e6
        else:
            objCCC[np.abs(objCCC) < 1e-8] = 0 # Set small values to zero
            
            objCCC_row = args[0].getFloatingSpeciesIds()
            objCCC_col = args[0].getReactionIds()
            objCCC = objCCC[np.argsort(objCCC_row)]
            objCCC = objCCC[:,np.argsort(objCCC_col)]
            
            dist_obj = ((np.linalg.norm(realConcCC - objCCC))*(1 + np.sum(np.not_equal(np.sign(realConcCC), np.sign(objCCC)))))
    except:
        countf += 1
        dist_obj = 1e6
        
    counts += 1
    
    return dist_obj


def updateTC(n, dists, Settings):
    terminate = False
    
    if (Settings.n_gen != None) and (n >= Settings.n_gen):
        terminate = True
    if ((Settings.gen_static != None) and (len(dists[3]) > Settings.gen_static) and 
        (np.all(dists[3][-Settings.gen_static:] == dists[3][-1]))):
        terminate = True
    if (Settings.thres_avg != None) and (dists[1][-1] <= Settings.thres_avg):
        terminate = True
    if (Settings.thres_median != None) and (dists[2][-1] <= Settings.thres_median):
        terminate = True
    if (Settings.thres_shortest != None) and (dists[0][-1] <= Settings.thres_shortest):
        terminate = True
    if (Settings.thres_top != None) and (dists[3][-1] <= Settings.thres_top):
        terminate = True
    
    return terminate


def callbackF(X, convergence=0.):
    global counts
    global countf
    print("{}, {}".format(counts,countf))
    return False

def mutate_and_evaluate(ens_model, ens_dist, ens_rl, ens_concCC, minind, mutind, Settings):

    global countf
    global counts
    global tracking
    
    eval_dist = np.empty(Settings.mut_size)
    eval_model = np.empty(Settings.mut_size, dtype='object')
    eval_rl = np.empty(Settings.mut_size, dtype='object')
    eval_concCC = np.empty(Settings.mut_size, dtype='object')
    
    mut_model = ens_model[mutind]
    mut_dist = ens_dist[mutind]
    mut_rl = ens_rl[mutind]
    mut_concCC = ens_concCC[mutind]
    
    for m in mut_range:
        cFalse = (1 + np.sum(np.not_equal(np.sign(realConcCC), 
                                          np.sign(mut_concCC[m])), axis=0))

        tempdiff = cFalse*np.linalg.norm(realConcCC - mut_concCC[m], axis=0)
        
        minrndidx = np.random.choice(minind)
        
        # stt = np.zeros((2,2))
        fid = []
        bid = []
        # stt = tracking[0]
        
        o = 0
        
        mutate_condition = True
        
        while mutate_condition:
            r_idx = np.random.choice(np.arange(nr), p=np.divide(tempdiff, np.sum(tempdiff)))
            
            reactionList = copy.deepcopy(mut_rl[m])
            
            if np.random.random() < Settings.recomb:
                reactionList[r_idx] = ens_rl[minrndidx][r_idx]
            else:
                ssum = np.sum(np.sign(realConcCC[:,r_idx]))
                if ssum > 0:
                    posRctInd = np.append(realFloatingIdsIndSort[realConcCC[:,r_idx] <= 0], 
                                          realBoundaryIdsIndSort)
                    posPrdInd = np.append(realFloatingIdsIndSort[realConcCC[:,r_idx] > 0], 
                                          realBoundaryIdsIndSort)
                elif ssum < 0:
                    posRctInd = np.append(realFloatingIdsIndSort[realConcCC[:,r_idx] < 0], 
                                          realBoundaryIdsIndSort)
                    posPrdInd = np.append(realFloatingIdsIndSort[realConcCC[:,r_idx] >= 0], 
                                          realBoundaryIdsIndSort)
                else:
                    posRctInd = np.append(realFloatingIdsIndSort[realConcCC[:,r_idx] <= 0], 
                                          realBoundaryIdsIndSort)
                    posPrdInd = np.append(realFloatingIdsIndSort[realConcCC[:,r_idx] >= 0], 
                                          realBoundaryIdsIndSort)
                    
                rct = [col[3] for col in reactionList]
                prd = [col[4] for col in reactionList]
                
                rType, regType, revType = ng.pickReactionType()
                
                # TODO: pick reactants and products based on control coefficients
                if rType == ng.ReactionType.UNIUNI:
                    # UniUni
                    rct_id = np.random.choice(posRctInd, size=1).tolist()
                    prd_id = np.random.choice(posPrdInd, size=1).tolist()
                    all_rct = [i for i,x in enumerate(rct) if x==rct_id]
                    all_prd = [i for i,x in enumerate(prd) if x==prd_id]
                    
                    while (((np.any(np.isin(rct_id, realBoundaryIdsInd))) and 
                           (np.any(np.isin(prd_id, realBoundaryIdsInd)))) or 
                           (len(set(all_rct) & set(all_prd)) > 0)):
                        rct_id = np.random.choice(posRctInd, size=1).tolist()
                        prd_id = np.random.choice(posPrdInd, size=1).tolist()
                        # Search for potentially identical reactions
                        all_rct = [i for i,x in enumerate(rct) if x==rct_id]
                        all_prd = [i for i,x in enumerate(prd) if x==prd_id]
                elif rType == ng.ReactionType.BIUNI:
                    # BiUni
                    rct_id = np.random.choice(posRctInd, size=2, replace=True).tolist()
                    prd_id = np.random.choice(posPrdInd, size=1).tolist()
                    all_rct = [i for i,x in enumerate(rct) if set(x)==set(rct_id)]
                    all_prd = [i for i,x in enumerate(prd) if x==prd_id]
                    
                    while (((np.any(np.isin(rct_id, realBoundaryIdsInd))) and 
                           (np.any(np.isin(prd_id, realBoundaryIdsInd)))) or 
                           (len(set(all_rct) & set(all_prd)) > 0)):
                        rct_id = np.random.choice(posRctInd, size=2, replace=True).tolist()
                        prd_id = np.random.choice(posPrdInd, size=1).tolist()
                        # Search for potentially identical reactions
                        all_rct = [i for i,x in enumerate(rct) if set(x)==set(rct_id)]
                        all_prd = [i for i,x in enumerate(prd) if x==prd_id]
                elif rType == ng.ReactionType.UNIBI:
                    # UniBi
                    rct_id = np.random.choice(posRctInd, size=1).tolist()
                    prd_id = np.random.choice(posPrdInd, size=2, replace=True).tolist()
                    all_rct = [i for i,x in enumerate(rct) if x==rct_id]
                    all_prd = [i for i,x in enumerate(prd) if set(x)==set(prd_id)]
                    
                    while (((np.any(np.isin(rct_id, realBoundaryIdsInd))) and 
                           (np.any(np.isin(prd_id, realBoundaryIdsInd)))) or 
                           (len(set(all_rct) & set(all_prd)) > 0)):
                        rct_id = np.random.choice(posRctInd, size=1).tolist()
                        prd_id = np.random.choice(posPrdInd, size=2, replace=True).tolist()
                        # Search for potentially identical reactions
                        all_rct = [i for i,x in enumerate(rct) if x==rct_id]
                        all_prd = [i for i,x in enumerate(prd) if set(x)==set(prd_id)]
                else:
                    # BiBi
                    rct_id = np.random.choice(posRctInd, size=2, replace=True).tolist()
                    prd_id = np.random.choice(posPrdInd, size=2, replace=True).tolist()
                    all_rct = [i for i,x in enumerate(rct) if set(x)==set(rct_id)]
                    all_prd = [i for i,x in enumerate(prd) if set(x)==set(prd_id)]
                    
                    while (((np.any(np.isin(rct_id, realBoundaryIdsInd))) and 
                           (np.any(np.isin(prd_id, realBoundaryIdsInd)))) or
                           (len(set(all_rct) & set(all_prd)) > 0)):
                        rct_id = np.random.choice(posRctInd, size=2, replace=True).tolist()
                        prd_id = np.random.choice(posPrdInd, size=2, replace=True).tolist()
                        # Search for potentially identical reactions
                        all_rct = [i for i,x in enumerate(rct) if set(x)==set(rct_id)]
                        all_prd = [i for i,x in enumerate(prd) if set(x)==set(prd_id)]
                        
                if regType == ng.RegulationType.DEFAULT:
                    act_id = []
                    inhib_id = []
                elif regType == ng.RegulationType.INHIBITION:
                    act_id = []
                    delList = np.concatenate([rct_id, prd_id])
                    if len(realBoundaryIdsInd) > 0:
                        delList = np.unique(np.append(delList, realBoundaryIdsInd))
                    cList = np.delete(nsList, delList)
                    if len(cList) == 0:
                        inhib_id = []
                        regType = ng.RegulationType.DEFAULT
                    else:
                        inhib_id = np.random.choice(cList, size=1).tolist()
                elif regType == ng.RegulationType.ACTIVATION:
                    inhib_id = []
                    delList = np.concatenate([rct_id, prd_id])
                    if len(realBoundaryIdsInd) > 0:
                        delList = np.unique(np.append(delList, realBoundaryIdsInd))
                    cList = np.delete(nsList, delList)
                    if len(cList) == 0:
                        act_id = []
                        regType = ng.RegulationType.DEFAULT
                    else:
                        act_id = np.random.choice(cList, size=1).tolist()
                else:
                    delList = np.concatenate([rct_id, prd_id])
                    if len(realBoundaryIdsInd) > 0:
                        delList = np.unique(np.append(delList, realBoundaryIdsInd))
                    cList = np.delete(nsList, delList)
                    if len(cList) < 2:
                        act_id = []
                        inhib_id = []
                        regType = ng.RegulationType.DEFAULT
                    else:
                        reg_id = np.random.choice(cList, size=2, replace=False)
                        act_id = [reg_id[0]]
                        inhib_id = [reg_id[1]]
                    
                reactionList[r_idx] = [rType, 
                                       regType, 
                                       revType, 
                                       rct_id, 
                                       prd_id, 
                                       act_id, 
                                       inhib_id]
            
            st = ng.getFullStoichiometryMatrix(reactionList, ns)
            stt, fid, bid = ng.removeBoundaryNodes(st, nsList, nrList)
            o += 1
            
            if Settings.trackStoichiometry:
                if ((fid != realFloatingIdsIndSortList or bid != realBoundaryIdsIndSortList 
                     or stt.tolist() in tracking or np.sum(stt) != 0 
                     or len(check_duplicate_reaction(stt)) > 0 or np.linalg.matrix_rank(stt) != realNumFloating) 
                    and (o < Settings.maxIter_mut)):
                    mutate_condition = True
                else:
                    mutate_condition = False
            else:
                if ((fid != realFloatingIdsIndSortList or bid != realBoundaryIdsIndSortList 
                     or reactionList in tracking or np.sum(stt) != 0 
                     or len(check_duplicate_reaction(stt)) > 0 or np.linalg.matrix_rank(stt) != realNumFloating) 
                    and (o < Settings.maxIter_mut)):
                    mutate_condition = True
                else:
                    mutate_condition = False
            
        
        if o >= Settings.maxIter_mut:
            eval_dist[m] = mut_dist[m]
            eval_model[m] = mut_model[m]
            eval_rl[m] = mut_rl[m]
            eval_concCC[m] = mut_concCC[m]
        else:
            antStr = ng.generateAntimony(realFloatingIdsSort, realBoundaryIdsSort, 
                                         fid, bid, reactionList, 
                                         boundary_init=realBoundaryVal)
            try:
                r = te.loada(antStr)
                concCC = customGetScaledConcentrationControlCoefficientMatrix(r)
                
                counts = 0
                countf = 0
                
                p_bound = ng.generateParameterBoundary(r.getGlobalParameterIds())
                res = scipy.optimize.differential_evolution(f1, args=(r,), 
                            bounds=p_bound, maxiter=Settings.optiMaxIter, 
                            tol=Settings.optiTol, polish=Settings.optiPolish, 
                            seed=Settings.r_seed)
                
                if not res.success or res.fun >= 1e6:
                    eval_dist[m] = mut_dist[m]
                    eval_model[m] = mut_model[m]
                    eval_rl[m] = mut_rl[m]
                    eval_concCC[m] = mut_concCC[m]
                else:
                    if res.fun < mut_dist[m]:
                        r.resetToOrigin()
                        r.setValues(r.getGlobalParameterIds(), res.x)
                        concCC = customGetScaledConcentrationControlCoefficientMatrix(r)
                        concCC[np.abs(concCC) < 1e-8] = 0
                        
                        if np.isnan(concCC).any():
                            eval_dist[m] = mut_dist[m]
                            eval_model[m] = mut_model[m]
                            eval_rl[m] = mut_rl[m]
                            eval_concCC[m] = mut_concCC[m]
                        else:
                            eval_dist[m] = res.fun
                            eval_model[m] = r.getAntimony(current=True)
                            eval_rl[m] = reactionList
                            if Settings.trackStoichiometry:
                                tracking.append(stt.tolist())
                            else:
                                tracking.append(reactionList)
                            eval_concCC[m] = concCC
                    else:
                        eval_dist[m] = mut_dist[m]
                        eval_model[m] = mut_model[m]
                        eval_rl[m] = mut_rl[m]
                        eval_concCC[m] = mut_concCC[m]
            except:
                eval_dist[m] = mut_dist[m]
                eval_model[m] = mut_model[m]
                eval_rl[m] = mut_rl[m]
                eval_concCC[m] = mut_concCC[m]
        
            try:
                r.clearModel()
            except:
                pass
            antimony.clearPreviousLoads()
            antimony.freeAll()

    return [eval_dist, eval_model, eval_rl, eval_concCC]


def initialize(Settings):

    global countf
    global counts
    global tracking
    
    numBadModels = 0
    numGoodModels = 0
    numIter = 0
    
    ens_dist = np.empty(Settings.ens_size)
    ens_model = np.empty(Settings.ens_size, dtype='object')
    ens_rl = np.empty(Settings.ens_size, dtype='object')
    tracking = []
    ens_concCC = np.empty(Settings.ens_size, dtype='object')
    
    # Initial Random generation
    while (numGoodModels < Settings.ens_size):
        rl = ng.generateReactionList(nsList, nrList, realFloatingIdsIndSort, 
                                     realBoundaryIdsIndSort, realConcCC)
        st = ng.getFullStoichiometryMatrix(rl, ns)
        stt, fid, bid = ng.removeBoundaryNodes(st, nsList, nrList)
        # Ensure no redundant model
        if Settings.trackStoichiometry:
            while (fid != realFloatingIdsIndSortList or bid != realBoundaryIdsIndSortList
                   or stt.tolist() in tracking or np.sum(stt) != 0 
                   or np.linalg.matrix_rank(stt) != realNumFloating):
                rl = ng.generateReactionList(nsList, nrList, realFloatingIdsIndSort, 
                                             realBoundaryIdsIndSort, realConcCC)
                st = ng.getFullStoichiometryMatrix(rl, ns)
                stt, fid, bid = ng.removeBoundaryNodes(st, nsList, nrList)
        else:
            while (fid != realFloatingIdsIndSortList or bid != realBoundaryIdsIndSortList
                   or rl in tracking or np.sum(stt) != 0 
                   or np.linalg.matrix_rank(stt) != realNumFloating):
                rl = ng.generateReactionList(nsList, nrList, realFloatingIdsIndSort, 
                                             realBoundaryIdsIndSort, realConcCC)
                st = ng.getFullStoichiometryMatrix(rl, ns)
                stt, fid, bid = ng.removeBoundaryNodes(st, nsList, nrList)
        antStr = ng.generateAntimony(realFloatingIdsSort, realBoundaryIdsSort, 
                                     fid, bid, rl, boundary_init=realBoundaryVal)

        try:
            r = te.loada(antStr)
            concCC = customGetScaledConcentrationControlCoefficientMatrix(r)
            
            counts = 0
            countf = 0
            
            p_bound = ng.generateParameterBoundary(r.getGlobalParameterIds())
            res = scipy.optimize.differential_evolution(f1, args=(r,), 
                                bounds=p_bound, maxiter=Settings.optiMaxIter, 
                                tol=Settings.optiTol, polish=Settings.optiPolish, 
                                seed=Settings.r_seed)
            
            if not res.success or res.fun >= 1e6:
                numBadModels += 1
            else:
                r.resetToOrigin()
                r.setValues(r.getGlobalParameterIds(), res.x)
                ens_dist[numGoodModels] = res.fun
                ens_model[numGoodModels] = r.getAntimony(current=True)
                ens_rl[numGoodModels] = rl
                if Settings.trackStoichiometry:
                    tracking.append(stt.tolist())
                else:
                    tracking.append(rl)
                concCC = customGetScaledConcentrationControlCoefficientMatrix(r)
                concCC[np.abs(concCC) < 1e-8] = 0
                ens_concCC[numGoodModels] = concCC
                
                numGoodModels = numGoodModels + 1
        except:
            numBadModels = numBadModels + 1
        
        numIter = numIter + 1
        if int(numIter/1000) == (numIter/1000):
            print("Number of iterations = {}".format(numIter))
        if int(numIter/10000) == (numIter/10000):
            print("Number of good models = {}".format(numGoodModels))
    
        try:
            r.clearModel()
        except:
            pass
        antimony.clearPreviousLoads()
        antimony.freeAll()
    
    print("In generation: 0")
    print("Number of total iterations = {}".format(numIter))
    print("Number of bad models = {}".format(numBadModels))
    
    return ens_dist, ens_model, ens_rl, tracking, ens_concCC


def random_gen(ens_model, ens_dist, ens_rl, ens_concCC, mut_ind_inv, Settings):
    
    global countf
    global counts
    global tracking
    
    listAntStr = ens_model[mut_ind_inv]
    listDist = ens_dist[mut_ind_inv]
    listrl = ens_rl[mut_ind_inv]
    listconcCC = ens_concCC[mut_ind_inv]
    
    rndSize = len(listDist)
    
    rnd_dist = np.empty(rndSize)
    rnd_model = np.empty(rndSize, dtype='object')
    rnd_rl = np.empty(rndSize, dtype='object')
    rnd_concCC = np.empty(rndSize, dtype='object')
    
    for l in range(rndSize):
        d = 0
        rl = ng.generateReactionList(nsList, nrList, realFloatingIdsIndSort, 
                                     realBoundaryIdsIndSort, realConcCC)
        st = ng.getFullStoichiometryMatrix(rl, ns)
        stt, fid, bid = ng.removeBoundaryNodes(st, nsList, nrList)
        # Ensure no redundant models
        if Settings.trackStoichiometry:
            while ((fid != realFloatingIdsIndSortList or bid != realBoundaryIdsIndSortList or 
                    stt.tolist() in tracking or np.sum(stt) != 0 or 
                    np.linalg.matrix_rank(stt) != realNumFloating) and (d < Settings.maxIter_gen)):
                rl = ng.generateReactionList(nsList, nrList, realFloatingIdsIndSort, 
                                             realBoundaryIdsIndSort, realConcCC)
                st = ng.getFullStoichiometryMatrix(rl, ns)
                stt, fid, bid = ng.removeBoundaryNodes(st, nsList, nrList)
                
                d += 1
        else:
            while ((fid != realFloatingIdsIndSortList or bid != realBoundaryIdsIndSortList or 
                    rl in tracking or np.sum(stt) != 0 or 
                    np.linalg.matrix_rank(stt) != realNumFloating) and (d < Settings.maxIter_gen)):
                rl = ng.generateReactionList(nsList, nrList, realFloatingIdsIndSort, 
                                             realBoundaryIdsIndSort, realConcCC)
                st = ng.getFullStoichiometryMatrix(rl, ns)
                stt, fid, bid = ng.removeBoundaryNodes(st, nsList, nrList)
                
                d += 1
        
        if d >= Settings.maxIter_gen:
            rnd_dist[l] = listDist[l]
            rnd_model[l] = listAntStr[l]
            rnd_rl[l] = listrl[l]
            rnd_concCC[l] = listconcCC[l]
        else:
            antStr = ng.generateAntimony(realFloatingIdsSort, realBoundaryIdsSort, 
                                         fid, bid, rl, boundary_init=realBoundaryVal)
            try:
                r = te.loada(antStr)
                concCC = customGetScaledConcentrationControlCoefficientMatrix(r)
                
                counts = 0
                countf = 0
                
                p_bound = ng.generateParameterBoundary(r.getGlobalParameterIds())
                res = scipy.optimize.differential_evolution(f1, args=(r,), 
                            bounds=p_bound, maxiter=Settings.optiMaxIter, 
                            tol=Settings.optiTol, polish=Settings.optiPolish, 
                            seed=Settings.r_seed)
                
                # Failed to find solution
                if not res.success or res.fun >= 1e6 or np.isnan(concCC).any():
                    rnd_dist[l] = listDist[l]
                    rnd_model[l] = listAntStr[l]
                    rnd_rl[l] = listrl[l]
                    rnd_concCC[l] = listconcCC[l]
                else:
                    if res.fun < listDist[l]:
                        r.resetToOrigin()
                        r.setValues(r.getGlobalParameterIds(), res.x)
                        rnd_dist[l] = res.fun
                        rnd_model[l] = r.getAntimony(current=True)
                        rnd_rl[l] = rl
                        if Settings.trackStoichiometry:
                            tracking.append(stt.tolist())
                        else:
                            tracking.append(rl)
                        concCC = customGetScaledConcentrationControlCoefficientMatrix(r)
                        concCC[np.abs(concCC) < 1e-8] = 0
                        rnd_concCC[l] = concCC
                    else:
                        rnd_dist[l] = listDist[l]
                        rnd_model[l] = listAntStr[l]
                        rnd_rl[l] = listrl[l]
                        rnd_concCC[l] = listconcCC[l]
            except:
                rnd_dist[l] = listDist[l]
                rnd_model[l] = listAntStr[l]
                rnd_rl[l] = listrl[l]
                rnd_concCC[l] = listconcCC[l]
        
            try:
                r.clearModel()
            except:
                pass        
            antimony.clearPreviousLoads()
            antimony.freeAll()
            
    return [rnd_dist, rnd_model, rnd_rl, rnd_concCC]


def argparse(argv):
    
    try:
        opts, args = getopt.getopt(argv, "hs:m:", ["help","setting=","model="])
    except getopt.GetoptError:
        print('main.py -s <settingfile> -m <modelfile>')
        sys.exit(2)
        
    for opt, arg in opts:
        if opt=="-h" or opt=="--help":
            print('main.py -s <settingfile> -m <modelfile>')
            sys.exit()
        elif opt in ("-s", "--setting"):
            Settings.READ_SETTINGS = arg
        elif opt in ("-m", "--model"):
            Settings.MODEL_INPUT = arg


# TODO: simulated annealing (multiply to fitness for rate constants)
if __name__ == '__main__':
#%% Import Settings
    Settings = SettingsClass()
    argparse(sys.argv[1:])
    
    if Settings.READ_SETTINGS != None:
        ioutils.readSettings(Settings)
    
    ioutils.validateSettings(Settings)
    
    Settings.EXPORT_PATH = ioutils.exportPathHandler(Settings)
    
#%% Analyze True Model
    roadrunner.Logger.disableLogging()
    # roadrunner.Config.setValue(roadrunner.Config.ROADRUNNER_DISABLE_WARNINGS, 3)
    
    # Restore
    rr_nmval = roadrunner.Config.getValue(roadrunner.Config.PYTHON_ENABLE_NAMED_MATRIX)

    # if conservedMoiety:
    #     roadrunner.Config.setValue(roadrunner.Config.LOADSBMLOPTIONS_CONSERVED_MOIETIES, True)
    
    # roadrunner.Config.setValue(roadrunner.Config.STEADYSTATE_APPROX, True)
    # roadrunner.Config.setValue(roadrunner.Config.STEADYSTATE_APPROX_MAX_STEPS, 5)
    # roadrunner.Config.setValue(roadrunner.Config.STEADYSTATE_APPROX_TIME, 10000)
    # roadrunner.Config.setValue(roadrunner.Config.STEADYSTATE_APPROX_TOL, 1e-3)
    # roadrunner.Config.setValue(roadrunner.Config.ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES, 1)
    roadrunner.Config.setValue(roadrunner.Config.PYTHON_ENABLE_NAMED_MATRIX, 0)
    # roadrunner.Config.setValue(roadrunner.Config.MAX_OUTPUT_ROWS, 5)

    if Settings.MODEL_INPUT != None:
        # Using custom models
        readModel = Settings.MODEL_INPUT
        try:
            realRR = te.loada(readModel)
            realModel = readModel
            realRL = ng.generateReactionListFromAntimony(realModel)
        except:
            try:
                realRR = te.loads(readModel)
                realModel = realRR.getAntimony()
                realRL = ng.generateReactionListFromAntimony(realModel)
            except:
                raise Exception("Cannot read the given model")
        
    else:
        # Using one of the test models
        realModel = ioutils.testModels(Settings.modelType)
        realRR = te.loada(realModel)
        realRL = ng.generateReactionListFromAntimony(realModel)
    
    # Species
    realNumFloating = realRR.getNumFloatingSpecies()
    realFloatingIds = realRR.getFloatingSpeciesIds()
    realFloatingIdsSort = np.sort(realFloatingIds)
    realFloatingIdsInd = np.fromiter(map(int, [s.strip('S') for s in realFloatingIds]), dtype=int)
    realFloatingIdsIndSort = np.sort(realFloatingIdsInd)
    realFloatingIdsIndSortList = list(realFloatingIdsIndSort)
    
    realNumBoundary = realRR.getNumBoundarySpecies()
    if realNumBoundary == 0:
        realNumBoundary = 1
        realBoundaryIds = ['S{}'.format(realNumFloating)]
        realBoundaryVal = [1]
    else:
        realBoundaryIds = realRR.getBoundarySpeciesIds()
        realBoundaryVal = realRR.getBoundarySpeciesConcentrations()
    realBoundaryIdsSort = np.sort(realBoundaryIds)
    realBoundaryIdsInd = np.fromiter(map(int, [s.strip('S') for s in realBoundaryIds]), dtype=int)
    realBoundaryIdsIndSort = np.sort(realBoundaryIdsInd)
    realBoundaryIdsIndSortList = list(realBoundaryIdsIndSort)
    
    realReactionIds = realRR.getReactionIds()
    realGlobalParameterIds = realRR.getGlobalParameterIds()
    
    # Control Coefficients and Fluxes
    re = realRR.simulate(0, 10000, 5)
    realSteadyState = realRR.getFloatingSpeciesConcentrations()
    realSteadyStateRatio = np.divide(realSteadyState, np.min(realSteadyState))
    realFlux = realRR.getReactionRates()
    realRR.reset()
    re = realRR.simulate(0, 10000, 5)
    realFluxCC = realRR.getScaledFluxControlCoefficientMatrix()
    realConcCC = realRR.getScaledConcentrationControlCoefficientMatrix()
    
    realFluxCC[np.abs(realFluxCC) < 1e-8] = 0
    realConcCC[np.abs(realConcCC) < 1e-8] = 0
    
    # Ordering
    realFluxCC = realFluxCC[np.argsort(realFloatingIds)]
    realFluxCC = realFluxCC[:,np.argsort(realReactionIds)]
    
    realConcCC = realConcCC[np.argsort(realFloatingIds)]
    realConcCC = realConcCC[:,np.argsort(realReactionIds)]
    
    realFlux = realFlux[np.argsort(realReactionIds)]
    
    # Number of Species and Ranges
    ns = realNumBoundary + realNumFloating # Number of species
    nsList = np.arange(ns)
    nr = realRR.getNumReactions() # Number of reactions
    nrList = np.arange(nr)
    
    ens_range = range(Settings.ens_size)
    Settings.mut_size = int(Settings.ens_size/2)
    mut_range = range(Settings.mut_size)
    
    realCount = np.array(np.unravel_index(np.argsort(realFluxCC, axis=None), realFluxCC.shape)).T
        
    #%%
    print("Original Control Coefficients")
    print(realConcCC)
    print("Original Steady State Ratio")
    print(realSteadyStateRatio)
    
    # Define seed and add noise
    np.random.seed(Settings.r_seed)
    
    if Settings.NOISE:
        for i in range(len(realConcCC)):
            for j in range(len(realConcCC[i])):
                realConcCC[i][j] = (realConcCC[i][j] + 
                          np.random.normal(0,Settings.ABS_NOISE_STD) +
                          np.random.normal(0,np.abs(realConcCC[i][j]*Settings.REL_NOISE_STD)))
        
        print("Control Coefficients with Noise Added")
        print(realConcCC)
    
    if Settings.RUN:
        process = psutil.Process()
        
        # Initualize Lists
        best_dist = []
        avg_dist = []
        med_dist = []
        top_dist = []
        
        # Start measuring time...
        t1 = time.time()
        
        memory = []
        memory.append(process.memory_info().rss)
        
        # Initialize
        ens_dist, ens_model, ens_rl, tracking, ens_concCC = initialize(Settings)
        
        memory.append(process.memory_info().rss)
        
        dist_top_ind = np.argsort(ens_dist)
        dist_top = ens_dist[dist_top_ind]
        model_top = ens_model[dist_top_ind]
        concCC_top = ens_concCC[dist_top_ind]
        
        best_dist.append(dist_top[0])
        avg_dist.append(np.average(dist_top))
        med_dist.append(np.median(dist_top))
        top_ind = int(Settings.top_p*Settings.ens_size)
        if top_ind == 0:
            top_ind = 1
        top_dist.append(np.average(np.unique(dist_top)[:top_ind]))
        
        print("Minimum distance: {}".format(best_dist[-1]))
        print("Top {} distance: {}".format(int(Settings.top_p*100), top_dist[-1]))
        print("Average distance: {}".format(avg_dist[-1]))
        
        terminate = False
        n = 0
        
        ens_idx = np.arange(Settings.ens_size)
        
        # breakFlag = False
        
        # TODO: Remove for loop
        # TODO: Add polishing with fast optimizer 
        while not terminate:
            minind = np.argsort(ens_dist)[:Settings.pass_size]
            
            g1 = np.random.choice(ens_idx, size=int(Settings.ens_size/2), 
                                  replace=False)
            g2 = np.delete(ens_idx, g1)
            
            fitg1 = g1[ens_dist[g1] <= ens_dist[g2]]
            fitg2 = g2[ens_dist[g1] > ens_dist[g2]]
            
            mut_ind = np.append(fitg1, fitg2)
            mut_ind_inv = np.setdiff1d(ens_idx, mut_ind)
            
            evol_output = mutate_and_evaluate(ens_model, ens_dist, ens_rl,
                                              ens_concCC, minind, mut_ind,
                                              Settings)
            ens_model[mut_ind] = evol_output[1]
            ens_dist[mut_ind] = evol_output[0]
            ens_rl[mut_ind] = evol_output[2]
            ens_concCC[mut_ind] = evol_output[3]
            
            # for tt in range(len(mut_ind)):
            #     r = te.loada(ens_model[mut_ind[tt]])
            #     try:
            #         r.steadyState()
            #     except:
            #         print("Failure detacted at mutation: ", mut_ind[tt])
            #         print(np.sort(mut_ind))
            #         breakFlag = True
            #         break
            
            # if breakFlag:
            #     break
            
            rnd_output = random_gen(ens_model, ens_dist, ens_rl, ens_concCC,
                                    mut_ind_inv, Settings)
            ens_model[mut_ind_inv] = rnd_output[1]
            ens_dist[mut_ind_inv] = rnd_output[0]
            ens_rl[mut_ind_inv] = rnd_output[2]
            ens_concCC[mut_ind_inv] = rnd_output[3]
            
            dist_top_ind = np.argsort(ens_dist)
            dist_top = ens_dist[dist_top_ind]
            model_top = ens_model[dist_top_ind]
            
            best_dist.append(dist_top[0])
            avg_dist.append(np.average(dist_top))
            med_dist.append(np.median(dist_top))
            top_dist.append(np.average(np.unique(dist_top)[:top_ind]))
            
            memory.append(process.memory_info().rss)
            
            print("In generation: {}".format(n+1))
            print("Minimum distance: {}".format(best_dist[-1]))
            print("Top {} distance: {}".format(int(Settings.top_p*100), top_dist[-1]))
            print("Average distance: {}".format(avg_dist[-1]))
            
            n += 1
            terminate = updateTC(n, [best_dist, avg_dist, med_dist, top_dist], 
                                 Settings)
            
            # for tt in range(len(mut_ind_inv)):
            #     r = te.loada(ens_model[mut_ind_inv[tt]])
            #     try:
            #         r.steadyState()
            #     except:
            #         print("Failure detacted at random gen: ", mut_ind_inv[tt])
            #         print(np.sort(mut_ind_inv))
            #         breakFlag = True
            #         break
            
            # if breakFlag:
            #     break
            
            # Error check
            # if np.average(dist_top) > 10000:
            #     break
    
        # Check the run time
        t2 = time.time()
        print("Run time: {}".format(t2-t1))
        
#%%
        # Collect models
        
        kdeOutput = analysis.selectWithKernalDensity(model_top, dist_top)
        
        if Settings.EXPORT_ALL_MODELS:
            model_col = model_top
            dist_col = dist_top
        else:
            model_col = model_top[:kdeOutput[0]]
            dist_col = dist_top[:kdeOutput[0]]
            
#%%
        # Settings.EXPORT_PATH = os.path.abspath(os.path.join(os.getcwd(), Settings.EXPORT_PATH))
        
        if Settings.SAVE_PLOT:
            if not os.path.exists(Settings.EXPORT_PATH):
                os.mkdir(Settings.EXPORT_PATH)
            if not os.path.exists(os.path.join(Settings.EXPORT_PATH, 'images')):
                os.mkdir(os.path.join(Settings.EXPORT_PATH, 'images'))
            pt.plotAllProgress([best_dist, avg_dist, med_dist, top_dist], show=Settings.SHOW_PLOT,
                               labels=['Best', 'Avg', 'Median', 'Top {} percent'.format(int(Settings.top_p*100))],
                               SAVE_PATH=os.path.join(Settings.EXPORT_PATH, 'images/AllConvergences.pdf'))
            pt.plotMemoryUsage(memory, show=Settings.SHOW_PLOT,
                               SAVE_PATH=os.path.join(Settings.EXPORT_PATH, 'images/memoryUsage.pdf'))
            pt.plotDistanceHistogramWithKDE(kdeOutput, dist_top, show=Settings.SHOW_PLOT,
                                            SAVE_PATH=os.path.join(Settings.EXPORT_PATH, 'images/distance_hist_w_KDE.pdf'))
        else:
            if Settings.SHOW_PLOT:
                pt.plotAllProgress([best_dist, avg_dist, med_dist, top_dist], 
                                   labels=['Best', 'Avg', 'Median', 'Top {} percent'.format(int(Settings.top_p*100))])
                pt.plotMemoryUsage(memory)
                pt.plotDistanceHistogramWithKDE(kdeOutput, dist_top)
                
#%%
        if Settings.EXPORT_SETTINGS or Settings.EXPORT_OUTPUT:
            if Settings.EXPORT_SETTINGS:
                ioutils.exportSettings(Settings, path=Settings.EXPORT_PATH)
            
            if Settings.EXPORT_OUTPUT:
                ioutils.exportOutputs(model_col, dist_col, [best_dist, avg_dist, med_dist, top_dist], 
                                      Settings, t2-t1, tracking, n, path=Settings.EXPORT_PATH)

    
    # Restore config
    roadrunner.Config.setValue(roadrunner.Config.PYTHON_ENABLE_NAMED_MATRIX, rr_nmval)

