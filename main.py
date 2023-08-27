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
import warnings

#np.seterr(all='raise')

class SettingsClass:
    
    def __init__(self):
        # Input ===============================================================
        # Load experimental input or preconfigured settings
        
        # Path to a custom model (default: None)
        self.MODEL_INPUT = None
        #TODO Path to experimental data - not implemented (default: None)
        self.DATA_INPUT = None
        # Path to preconfigured settings (default: None)
        self.READ_SETTINGS = None
        # Path to a previous output (default: None)
        self.CACHED_DIR = None
        
        
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
        # Number of models used for recombination (default: int(0.1*ens_size))
        self.pass_size = int(self.ens_size/10)
        # Top percentage of population to track (default: 0.05)
        self.top_p = 0.1
        # Maximum iteration allowed for initialization (default: 10000)
        self.maxIter_init = 10000
        # Maximum iteration allowed for random generation (default: 200)
        self.maxIter_gen = 200
        # Maximum iteration allowed for mutation (default: 200)
        self.maxIter_mut = 200
        # Recombination probability (default: 0.3)
        self.recomb = 0.3
        # Set conserved moiety (default: False)
        self.conservedMoiety = False
        # When testing, check if the correst stoichiometry was recovered (default: True)
        self.checkCorrectStoichiometry = True
        # Prune unnecessary boundary species at the end (default: True)
        self.prune = True
        
        
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
        # Maximum run time allowed in minutes
        self.max_run_time = None
        
        
        # Optimizer settings ==================================================
        # Settings specific to differential evolution
        
        # Maximum iteration allowed (default: 1000)
        self.optiMaxIter = 1000
        # Optimizer tolerance (default: 1)
        self.optiTol = 1.
        # Allow polishing parameters for optimizer (default: False)
        self.optiPolish = False
        # Run optimization at the end for better fitness representation (default: False)
        self.refine = False
        # Tolerance for additional optimization at the end (default: 0.01)
        self.refineTol = 0.01
        # Weight for control coefficients when calculating the distance - unused
        self.w1 = 16
        # Weight for steady-state and flux when calculating the distance - unused
        self.w2 = 1.0
        
        
        # Reaction settings ===================================================
        # TODO: allow modification of probability
        
        
        # RNG and noise settings ==============================================
        
        # Seed
        self.r_seed = 123123
        # Flag to add Gaussian noise to the input
        self.NOISE = False
        # Standard deviation of absolute noise
        self.ABS_NOISE_STD = 0.005
        # Standard deviation of relative noise
        self.REL_NOISE_STD = 0.2
        
        
        # Plotting settings ===================================================
        
        # Flag to visualize plot
        self.SHOW_PLOT = True
        # Flag to save figures
        self.SAVE_PLOT = True
        
        
        # Export settings =====================================================
            
        # Flag to collect all models in the ensemble
        self.EXPORT_ALL_MODELS = True
        # Flag to save collected models
        self.EXPORT_OUTPUT = True
        # Flag to save current settings
        self.EXPORT_SETTINGS = True
        # Flag to save model components for caching
        self.EXPORT_CACHE = True
        # Path to save the output
        self.EXPORT_PATH = './outputs/prune'
        # Overwrite the contents if the folder exists
        self.EXPORT_OVERWRITE = False
        # Create folders based on model names
        self.EXPORT_FORCE_MODELNAMES = False
        
        
        # Flag to run the algorithm - temporary
        self.RUN = False


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
            objCCC[np.abs(objCCC) < 1e-7] = 0 # Set small values to zero
            
            if args[0].conservedMoietyAnalysis:
                objCCC = objCCC[np.argsort(args[0].getFloatingSpeciesIds())]
            
            dist_obj = ((np.linalg.norm(realConcCC - objCCC))*(1 + np.sum(np.not_equal(np.sign(realConcCC), np.sign(objCCC)))))
    except:
        countf += 1
        dist_obj = 1e6
        
    counts += 1
    
    return dist_obj


def updateTC(n, runtime, dists, Settings):
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
    if (Settings.max_run_time != None) and (runtime/60 >= Settings.max_run_time):
        terminate = True
    
    return terminate


def callbackF(X, convergence=0.):
    global counts
    global countf
    print("{}, {}".format(counts,countf))
    return False


def mutate_and_evaluate_stoich(Settings, ens_dist, ens_model, ens_stoi, ens_rtypes, 
                               ens_ia, ens_concCC, minind, mutind):
    global countf
    global counts
    global tracking
    
    eval_dist = np.empty(Settings.mut_size)
    eval_model = np.empty(Settings.mut_size, dtype='object')
    eval_stoi = np.empty((Settings.mut_size, ns, nr), dtype=int)
    eval_rtypes = np.empty((Settings.mut_size, 3, nr), dtype=int)
    eval_ia = np.empty((Settings.mut_size, ns, nr), dtype=int)
    eval_concCC = np.empty(Settings.mut_size, dtype='object')
    
    mut_dist = ens_dist[mutind]
    mut_model = ens_model[mutind]
    mut_stoi = ens_stoi[mutind]
    mut_rtypes = ens_rtypes[mutind]
    mut_ia = ens_ia[mutind]
    mut_concCC = ens_concCC[mutind]
    
    for m in mut_range:
        cFalse = (1 + np.sum(np.not_equal(realSigns, np.sign(mut_concCC[m])), axis=0))
        tempdiff = cFalse*np.linalg.norm(realConcCC - mut_concCC[m], axis=0)
        minrndidx = np.random.choice(minind)
        
        o = 0
        mutate_condition = True
        
        while mutate_condition:
            sttsum = 1
            sttrank = 0
            noprd = True
            norct = True
            alreadyexists = True
            dups = True
            
            try:
                r_idx = np.random.choice(nrList, p=np.divide(tempdiff, np.sum(tempdiff)))
                
                stoi = copy.deepcopy(mut_stoi[m])
                stoi[:,r_idx] = np.zeros(ns)
                rtypes = copy.deepcopy(mut_rtypes[m])
                ia = copy.deepcopy(mut_ia[m])
                
                if np.random.random() < Settings.recomb:
                    stoi[:,r_idx] = ens_stoi[minrndidx][:,r_idx]
                    rtypes[:,r_idx] = ens_rtypes[minrndidx][:,r_idx]
                    ia[:,r_idx] = ens_ia[minrndidx][:,r_idx]
                else:
                    stoi, rTyper, iar  = ng.generateSingleST(stoi, r_idx, realSigns, 
                                                             realFloatingIdsInd, 
                                                             realBoundaryIdsInd, ns, nr)
                    rtypes[0,r_idx] = rTyper
                    ia[:,r_idx] = iar
                    
                stt = stoi[realFloatingIdsInd]
                stt[stt > 1] = 1
                stt[stt < -1] = -1
                
                sttsum = np.sum(stt)
                sttrank = np.linalg.matrix_rank(stt)
                noprd = any(np.sum(stt>0, axis=1) == 0)
                norct = any(np.sum(stt<0, axis=1) == 0)
                alreadyexists = stt.tolist() in tracking
                dups = len(check_duplicate_reaction(st)) > 0
            except:
                pass
            
            o += 1
                
            if ((sttsum != 0 or sttrank != realNumFloating or
                   norct or noprd or alreadyexists or dups) and (o < Settings.maxIter_mut)):
                mutate_condition = True
            else:
                mutate_condition = False
        
        if o >= Settings.maxIter_mut:
            eval_dist[m] = mut_dist[m]
            eval_model[m] = mut_model[m]
            eval_stoi[m] = mut_stoi[m]
            eval_rtypes[m] = mut_rtypes[m]
            eval_ia[m] = mut_ia[m]
            eval_concCC[m] = mut_concCC[m]
        else:
            antStr = ng.generateAntimonyfromST(realFloatingIds, realBoundaryIds, 
                                               stoi, rtypes, ia, boundary_init=realBoundaryVal)
            try:
                r = te.loada(antStr)
                gpid = [x for x in r.getGlobalParameterIds() if not x.startswith('S')]
                concCC = analysis.customGetScaledConcentrationControlCoefficientMatrix(r)
                # concCC = r.getScaledConcentrationControlCoefficientMatrix()
                
                counts = 0
                countf = 0
                
                p_bound = ng.generateParameterBoundary(gpid)
                res = scipy.optimize.differential_evolution(f1, args=(r,), 
                            bounds=p_bound, maxiter=Settings.optiMaxIter, 
                            tol=Settings.optiTol, polish=Settings.optiPolish, 
                            seed=Settings.r_seed)
                
                if not res.success or res.fun >= 1e6:
                    eval_dist[m] = mut_dist[m]
                    eval_model[m] = mut_model[m]
                    eval_stoi[m] = mut_stoi[m]
                    eval_rtypes[m] = mut_rtypes[m]
                    eval_ia[m] = mut_ia[m]
                    eval_concCC[m] = mut_concCC[m]
                else:
                    if res.fun < mut_dist[m]:
                        r.resetToOrigin()
                        r.setValues(gpid, res.x)
                        concCC = analysis.customGetScaledConcentrationControlCoefficientMatrix(r)
                        # concCC = r.getScaledConcentrationControlCoefficientMatrix()
                        if np.isnan(concCC).any():
                            eval_dist[m] = mut_dist[m]
                            eval_model[m] = mut_model[m]
                            eval_stoi[m] = mut_stoi[m]
                            eval_rtypes[m] = mut_rtypes[m]
                            eval_ia[m] = mut_ia[m]
                            eval_concCC[m] = mut_concCC[m]
                        else:
                            concCC[np.abs(concCC) < 1e-7] = 0
                            if r.conservedMoietyAnalysis:
                                concCC = concCC[np.argsort(r.getFloatingSpeciesIds())]
                            eval_dist[m] = res.fun
                            r.reset()
                            eval_model[m] = r.getAntimony(current=True)
                            eval_stoi[m] = stoi
                            tracking.append(stt.tolist())
                            eval_rtypes[m] = rtypes
                            eval_ia[m] = ia
                            eval_concCC[m] = concCC
                    else:
                        eval_dist[m] = mut_dist[m]
                        eval_model[m] = mut_model[m]
                        eval_stoi[m] = mut_stoi[m]
                        eval_rtypes[m] = mut_rtypes[m]
                        eval_ia[m] = mut_ia[m]
                        eval_concCC[m] = mut_concCC[m]
            except:
                eval_dist[m] = mut_dist[m]
                eval_model[m] = mut_model[m]
                eval_stoi[m] = mut_stoi[m]
                eval_rtypes[m] = mut_rtypes[m]
                eval_ia[m] = mut_ia[m]
                eval_concCC[m] = mut_concCC[m]
        
            try:
                r.clearModel()
            except:
                pass
            antimony.clearPreviousLoads()
            antimony.freeAll()

    return (eval_dist, eval_model, eval_stoi, eval_rtypes, eval_ia, eval_concCC)


def initialize(Settings):
    global countf
    global counts
    global tracking
    
    print("Starting initialization...")
    
    numBadModels = 0
    numGoodModels = 0
    numIter = 0
    
    ens_dist = np.empty(Settings.ens_size)
    ens_model = np.empty(Settings.ens_size, dtype='object')
    ens_stoi = np.empty((Settings.ens_size, ns, nr), dtype=int)
    ens_rtypes = np.empty((Settings.ens_size, 3, nr), dtype=int)
    ens_ia = np.empty((Settings.ens_size, ns, nr), dtype=int)
    tracking = []
    ens_concCC = np.empty(Settings.ens_size, dtype='object')
    
    # Initial Random generation
    while (numGoodModels < Settings.ens_size):
        numGen = 0
        # Ensure no redundant model
        sttsum = 1
        sttrank = 0
        noprd = True
        norct = True
        alreadyexists = True
        dups = True
        
        while (sttsum != 0 or sttrank != realNumFloating or
               norct or noprd or alreadyexists or dups):
            try:
                st, stt, rTypes, ia = ng.generateST(realSigns, realFloatingIdsInd, 
                                                    realBoundaryIdsInd, ns, nr)
                sttsum = np.sum(stt)
                sttrank = np.linalg.matrix_rank(stt)
                noprd = any(np.sum(stt>0, axis=1) == 0)
                norct = any(np.sum(stt<0, axis=1) == 0)
                alreadyexists = stt.tolist() in tracking
                dups = len(check_duplicate_reaction(st)) > 0
            
                numGen += 1
                if int(numGen/1000) == (numGen/1000):
                    print("Number of init. model gen. iter. = {}".format(numGen))
                if numGen > Settings.maxIter_init:
                    raise Exception("Failed to initialize. Population size may be too large.")
            except:
                pass
        antStr = ng.generateAntimonyfromST(realFloatingIds, realBoundaryIds, 
                                           st, rTypes, ia, boundary_init=realBoundaryVal)
        try:
            r = te.loada(antStr)
            gpid = [x for x in r.getGlobalParameterIds() if not x.startswith('S')]
            concCC = analysis.customGetScaledConcentrationControlCoefficientMatrix(r)
            # concCC = r.getScaledConcentrationControlCoefficientMatrix()
            
            counts = 0
            countf = 0
            
            p_bound = ng.generateParameterBoundary(gpid)
            res = scipy.optimize.differential_evolution(f1, args=(r,), 
                                bounds=p_bound, maxiter=Settings.optiMaxIter, 
                                tol=Settings.optiTol, polish=Settings.optiPolish, 
                                seed=Settings.r_seed)
            
            if not res.success or res.fun >= 1e6:
                numBadModels += 1
            else:
                r.resetToOrigin()
                r.setValues(gpid, res.x)
                ens_dist[numGoodModels] = res.fun
                ens_model[numGoodModels] = r.getAntimony(current=True)
                ens_stoi[numGoodModels] = st
                tracking.append(stt.tolist())
                ens_rtypes[numGoodModels] = rTypes
                ens_ia[numGoodModels] = ia
                concCC = analysis.customGetScaledConcentrationControlCoefficientMatrix(r)
                # concCC = r.getScaledConcentrationControlCoefficientMatrix()
                concCC[np.abs(concCC) < 1e-7] = 0
                if r.conservedMoietyAnalysis:
                    concCC = concCC[np.argsort(r.getFloatingSpeciesIds())]
                ens_concCC[numGoodModels] = concCC
                
                numGoodModels += 1
                print(numGoodModels)
        except:
            numBadModels += 1
        
        numIter = numIter + 1
        if int(numIter/100) == (numIter/100):
            print("Number of iterations = {}".format(numIter))
        if int(numIter/100) == (numIter/100):
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
    
    return (ens_dist, ens_model, ens_stoi, ens_rtypes, ens_ia, ens_concCC, tracking)


def random_gen(Settings, ens_model, ens_dist, ens_stoi, ens_rtypes, 
               ens_ia, ens_concCC, mut_ind_inv):
    
    global countf
    global counts
    global tracking
    
    listAntStr = ens_model[mut_ind_inv]
    listDist = ens_dist[mut_ind_inv]
    liststoi = ens_stoi[mut_ind_inv]
    listrtypes = ens_rtypes[mut_ind_inv]
    listia = ens_ia[mut_ind_inv]
    listconcCC = ens_concCC[mut_ind_inv]
    
    rndSize = len(listDist)
    
    rnd_dist = np.empty(rndSize)
    rnd_model = np.empty(rndSize, dtype='object')
    rnd_stoi = np.empty((rndSize, ns, nr), dtype=int)
    rnd_rtypes = np.empty((rndSize, 3, nr), dtype=int)
    rnd_ia = np.empty((rndSize, ns, nr), dtype=int)
    rnd_concCC = np.empty(rndSize, dtype='object')
    
    for l in range(rndSize):
        d = 0
        # Ensure no redundant models
        sttsum = 1
        sttrank = 0
        noprd = True
        norct = True
        alreadyexists = True
        dups = True
        
        while (sttsum != 0 or sttrank != realNumFloating or
               norct or noprd or alreadyexists or dups) and (d < Settings.maxIter_gen):
            try:
                st, stt, rTypes, ia = ng.generateST(realSigns, realFloatingIdsInd, 
                                                    realBoundaryIdsInd, ns, nr)
                sttsum = np.sum(stt)
                sttrank = np.linalg.matrix_rank(stt)
                noprd = any(np.sum(stt>0, axis=1) == 0)
                norct = any(np.sum(stt<0, axis=1) == 0)
                alreadyexists = stt.tolist() in tracking
                dups = len(check_duplicate_reaction(st)) > 0
            except:
                pass
            d += 1
            
        if d >= Settings.maxIter_gen:
            rnd_dist[l] = listDist[l]
            rnd_model[l] = listAntStr[l]
            rnd_stoi[l] = liststoi[l]
            rnd_rtypes[l] = listrtypes[l]
            rnd_ia[l] = listia[l]
            rnd_concCC[l] = listconcCC[l]
        else:
            antStr = ng.generateAntimonyfromST(realFloatingIds, realBoundaryIds, 
                                               st, rTypes, ia, boundary_init=realBoundaryVal)
            try:
                r = te.loada(antStr)
                gpid = [x for x in r.getGlobalParameterIds() if not x.startswith('S')]
                concCC = analysis.customGetScaledConcentrationControlCoefficientMatrix(r)
                # concCC = r.getScaledConcentrationControlCoefficientMatrix()
                
                counts = 0
                countf = 0
                
                p_bound = ng.generateParameterBoundary(gpid)
                res = scipy.optimize.differential_evolution(f1, args=(r,), 
                            bounds=p_bound, maxiter=Settings.optiMaxIter, 
                            tol=Settings.optiTol, polish=Settings.optiPolish, 
                            seed=Settings.r_seed)
                
                # Failed to find solution
                if not res.success or res.fun >= 1e6 or np.isnan(concCC).any():
                    rnd_dist[l] = listDist[l]
                    rnd_model[l] = listAntStr[l]
                    rnd_stoi[l] = liststoi[l]
                    rnd_rtypes[l] = listrtypes[l]
                    rnd_ia[l] = listia[l]
                    rnd_concCC[l] = listconcCC[l]
                else:
                    if res.fun < listDist[l]:
                        r.resetToOrigin()
                        r.setValues(gpid, res.x)
                        rnd_dist[l] = res.fun
                        rnd_model[l] = r.getAntimony(current=True)
                        tracking.append(stt.tolist())
                        rnd_stoi[l] = st
                        rnd_rtypes[l] = rTypes
                        rnd_ia[l] = ia
                        concCC = analysis.customGetScaledConcentrationControlCoefficientMatrix(r)
                        # concCC = r.getScaledConcentrationControlCoefficientMatrix()
                        concCC[np.abs(concCC) < 1e-7] = 0
                        if r.conservedMoietyAnalysis:
                            concCC = concCC[np.argsort(r.getFloatingSpeciesIds())]
                        rnd_concCC[l] = concCC
                    else:
                        rnd_dist[l] = listDist[l]
                        rnd_model[l] = listAntStr[l]
                        rnd_stoi[l] = liststoi[l]
                        rnd_rtypes[l] = listrtypes[l]
                        rnd_ia[l] = listia[l]
                        rnd_concCC[l] = listconcCC[l]
            except:
                rnd_dist[l] = listDist[l]
                rnd_model[l] = listAntStr[l]
                rnd_stoi[l] = liststoi[l]
                rnd_rtypes[l] = listrtypes[l]
                rnd_ia[l] = listia[l]
                rnd_concCC[l] = listconcCC[l]
        
            try:
                r.clearModel()
            except:
                pass        
            antimony.clearPreviousLoads()
            antimony.freeAll()
            
    return (rnd_dist, rnd_model, rnd_stoi, rnd_rtypes, rnd_ia, rnd_concCC)


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
        elif opt in ("-c", "--cache"):
            Settings.CACHED_DIR = arg


if __name__ == '__main__':
#%% Import Settings
    Settings = SettingsClass()
    argparse(sys.argv[1:])
    
    if Settings.MODEL_INPUT != None:
        try:
            Settings.MODEL_INPUT = os.path.abspath(Settings.MODEL_INPUT)
        except:
            raise Exception("Cannot find the given model")
    
    if Settings.DATA_INPUT != None:
        try:
            Settings.DATA_INPUT = os.path.abspath(Settings.DATA_INPUT)
        except:
            raise Exception("Cannot find the given data")
    
    if Settings.CACHED_DIR != None:
        Settings.READ_SETTINGS = os.path.join(Settings.CACHED_DIR, 'settings.txt')
    
    if Settings.READ_SETTINGS != None:
        ioutils.readSettings(Settings)
    
    ioutils.validateSettings(Settings)
    
#%% Analyze True Model
    roadrunner.Logger.disableLogging()
    # roadrunner.Config.setValue(roadrunner.Config.ROADRUNNER_DISABLE_WARNINGS, 3)
    
    # Restore
    rr_nmval = roadrunner.Config.getValue(roadrunner.Config.PYTHON_ENABLE_NAMED_MATRIX)

    # if conservedMoiety:
    #     roadrunner.Config.setValue(roadrunner.Config.LOADSBMLOPTIONS_CONSERVED_MOIETIES, True)
    
    roadrunner.Config.setValue(roadrunner.Config.PYTHON_ENABLE_NAMED_MATRIX, 0)
    
    # roadrunner.Config.setValue(roadrunner.Config.STEADYSTATE_APPROX, True)
    # roadrunner.Config.setValue(roadrunner.Config.STEADYSTATE_APPROX_MAX_STEPS, 5)
    # roadrunner.Config.setValue(roadrunner.Config.STEADYSTATE_APPROX_TIME, 10000)
    # roadrunner.Config.setValue(roadrunner.Config.STEADYSTATE_APPROX_TOL, 1e-3)
    # roadrunner.Config.setValue(roadrunner.Config.ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES, 1)
    # roadrunner.Config.setValue(roadrunner.Config.MAX_OUTPUT_ROWS, 5)

    if Settings.MODEL_INPUT != None:
        # Using custom models
        readModel = Settings.MODEL_INPUT
        try:
            realRR = te.loada(readModel)
            realModel = readModel
            # realRL = ng.generateReactionListFromAntimony(realModel)
        except:
            try:
                realRR = te.loads(readModel)
                realModel = realRR.getAntimony()
                # realRL = ng.generateReactionListFromAntimony(realModel)
            except:
                raise Exception("Cannot read the given model")
        
    else:
        # Using one of the test models
        realModel = ioutils.testModels(Settings.modelType)
        realRR = te.loada(realModel)
        # realRL = ng.generateReactionListFromAntimony(realModel)
    
    # Control Coefficients and Fluxes
    re = realRR.simulate(0, 10000, 5)
    realSteadyState = realRR.getFloatingSpeciesConcentrations()
    realSteadyStateRatio = np.divide(realSteadyState, np.min(realSteadyState))
    realFlux = realRR.getReactionRates()
    realRR.reset()
    re = realRR.simulate(0, 10000, 5)
    realFluxCC = realRR.getScaledFluxControlCoefficientMatrix()
    realConcCC = realRR.getScaledConcentrationControlCoefficientMatrix()
    
    realFluxCC[np.abs(realFluxCC) < 1e-7] = 0
    realConcCC[np.abs(realConcCC) < 1e-7] = 0
    
    if realRR.conservedMoietyAnalysis:
        realConcCC = realConcCC[np.argsort(realRR.getFloatingSpeciesIds())]
    
    realSigns = np.sign(realConcCC).astype(int)
    
    # Species
    realNumFloating = realRR.getNumFloatingSpecies()
    
    realFloatingIds = []
    realFloatingIdsIndList = []
    for i in range(realNumFloating):
        realFloatingIds.append('S{}'.format(i))
        realFloatingIdsIndList.append(i)
    fid_dict = dict(zip(realFloatingIds, np.sort(realRR.getFloatingSpeciesIds())))
    
    realFloatingIdsInd = np.array(realFloatingIdsIndList)
    
    realNumBoundary = realRR.getNumBoundarySpecies()
    if realNumBoundary == 0:
        realNumBoundary = 1
        realBoundaryIds = ['S{}'.format(realNumFloating)]
        realBoundaryIdsIndList = [realNumFloating]
        bid_dict = dict(zip(realBoundaryIds, ['sink']))
        realBoundaryVal = [1]
    else:
        realBoundaryIds = []
        realBoundaryIdsIndList = []
        for i in range(realNumBoundary):
            realBoundaryIds.append('S{}'.format(i+realNumFloating))
            realBoundaryIdsIndList.append(i+realNumFloating)
        bid_dict = dict(zip(realBoundaryIds, realRR.getBoundarySpeciesIds()))
        realBoundaryVal = realRR.getBoundarySpeciesConcentrations()
        
    realBoundaryIdsInd = np.array(realBoundaryIdsIndList)
    
    # Stoichiometry
    realStoi = realRR.getFullStoichiometryMatrix().astype(int)

    # Number of Species and Ranges
    ns = realNumBoundary + realNumFloating # Number of species
    nsList = np.arange(ns)
    nr = realRR.getNumReactions() # Number of reactions
    nrList = np.arange(nr)
    
    ens_range = range(Settings.ens_size)
    Settings.mut_size = int(Settings.ens_size/2)
    mut_range = range(Settings.mut_size)
    
        
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
        
        # TODO: bundle dist stats in a single 2D array
        # Initialize fitness statistics
        if Settings.CACHED_DIR != None:
            (best_dist, avg_dist, med_dist, top_dist) = ioutils.readStats(Settings)
        else:
            best_dist = []
            avg_dist = []
            med_dist = []
            top_dist = []
        
        # Start measuring time...
        t1 = time.time()
        
        memory = []
        memory.append(process.memory_info().rss)
        
        # Initialize
        if Settings.CACHED_DIR != None:
            (ens_dist, ens_model, ens_stoi, ens_rtypes, ens_ia, ens_concCC, 
             tracking) = ioutils.readCache(Settings)
        else:
            (ens_dist, ens_model, ens_stoi, ens_rtypes, ens_ia, ens_concCC, 
             tracking) = initialize(Settings)

        memory.append(process.memory_info().rss)
        
        dist_top_ind = np.argsort(ens_dist)
        dist_top = ens_dist[dist_top_ind]
        model_top = ens_model[dist_top_ind]
        
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
        
        while not terminate:
            minind = np.argsort(ens_dist)[:Settings.pass_size]
            
            g1 = np.random.choice(ens_idx, size=int(Settings.ens_size/2), 
                                  replace=False)
            g2 = np.delete(ens_idx, g1)
            
            fitg1 = g1[ens_dist[g1] <= ens_dist[g2]]
            fitg2 = g2[ens_dist[g1] > ens_dist[g2]]
            
            mutind = np.append(fitg1, fitg2)
            mutind_inv = np.setdiff1d(ens_idx, mutind)
            
            evol_output = mutate_and_evaluate_stoich(Settings, ens_dist, ens_model, 
                                                     ens_stoi, ens_rtypes, ens_ia, 
                                                     ens_concCC, minind, mutind)
            ens_dist[mutind] = evol_output[0]
            ens_model[mutind] = evol_output[1]
            ens_stoi[mutind] = evol_output[2]
            ens_rtypes[mutind] = evol_output[3]
            ens_ia[mutind] = evol_output[4]
            ens_concCC[mutind] = evol_output[5]
            
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
            rnd_output = random_gen(Settings, ens_model, ens_dist, ens_stoi, 
                                    ens_rtypes, ens_ia, ens_concCC, mutind_inv)
            ens_dist[mutind_inv] = rnd_output[0]
            ens_model[mutind_inv] = rnd_output[1]
            ens_stoi[mutind_inv] = rnd_output[2]
            ens_rtypes[mutind_inv] = rnd_output[3]
            ens_ia[mutind_inv] = rnd_output[4]
            ens_concCC[mutind_inv] = rnd_output[5]
            
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
            runtime = time.time()-t1
            terminate = updateTC(n, runtime, [best_dist, avg_dist, med_dist, top_dist], 
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
    
        # Print the run time
        print("Run time: {}".format(runtime))
        
#%%
        if Settings.prune:
            bidx = np.arange(realNumFloating, realNumFloating+realNumBoundary)
            
            for i,j in enumerate(ens_model):
                st = ens_stoi[i]
                rTypes = ens_rtypes[i]
                ia = ens_ia[i]
                for k in range(len(st[0])):
                    if len(st) < realNumFloating + realNumBoundary:
                        pass
                    else:
                        if np.sum(st[:,k] > 0) > 1:
                            if np.sum(st[bidx,k] > 0) > 1:
                                ridx = np.random.choice(np.arange(realNumBoundary)[st[bidx,k] > 0])
                                st[bidx[st[bidx,k] > 0][ridx],k] = 0
                                if rTypes[0,k] == 3:
                                    rTypes[0,k] = 1
                                elif rTypes[0,k] == 2:
                                    rTypes[0,k] = 0
                            else:
                                if len(st[bidx[st[bidx,k] > 0],k]) > 0:
                                    st[bidx[st[bidx,k] > 0],k] = 0
                                    if rTypes[0,k] == 3:
                                        rTypes[0,k] = 1
                                    elif rTypes[0,k] == 2:
                                        rTypes[0,k] = 0
                            
                        if np.sum(st[:,k] < 0) > 1:
                            if np.sum(st[bidx,k] < 0) > 1:
                                ridx = np.random.choice(np.arange(realNumBoundary)[st[bidx,k] < 0])
                                st[bidx[st[bidx,k] < 0][ridx],k] = 0
                                if rTypes[0,k] == 3:
                                    rTypes[0,k] = 2
                                elif rTypes[0,k] == 1:
                                    rTypes[0,k] = 0
                            else:
                                if len(st[bidx[st[bidx,k] < 0],k]) > 0:
                                    st[bidx[st[bidx,k] < 0],k] = 0
                                    if rTypes[0,k] == 3:
                                        rTypes[0,k] = 2
                                    elif rTypes[0,k] == 1:
                                        rTypes[0,k] = 0
                    
                    if np.sum(st[realNumFloating:,k] < -1) > 0:
                        if rTypes[0,k] == 3:
                            rTypes[0,k] = 2
                        elif rTypes[0,k] == 1:
                            rTypes[0,k] = 0
                    
                    if np.sum(st[realNumFloating:,k] > 1) > 0:
                        if rTypes[0,k] == 3:
                            rTypes[0,k] = 1
                        elif rTypes[0,k] == 2:
                            rTypes[0,k] = 0
                    
                antStr = ng.generateAntimonyfromST(realFloatingIds, realBoundaryIds, 
                                                   st, rTypes, ia, boundary_init=realBoundaryVal)
                r = te.loada(antStr)
                gpid = [x for x in r.getGlobalParameterIds() if not x.startswith('S')]
                p_bound = ng.generateParameterBoundary(gpid)
                res = scipy.optimize.differential_evolution(f1, args=(r,), 
                                    bounds=p_bound, maxiter=Settings.optiMaxIter, 
                                    tol=Settings.optiTol, polish=Settings.optiPolish, 
                                    seed=Settings.r_seed)
            
                r.resetToOrigin()
                r.setValues(gpid, res.x)
                concCC = analysis.customGetScaledConcentrationControlCoefficientMatrix(r)
                # concCC = r.getScaledConcentrationControlCoefficientMatrix()
                concCC[np.abs(concCC) < 1e-7] = 0
                if r.conservedMoietyAnalysis:
                    concCC = concCC[np.argsort(r.getFloatingSpeciesIds())]
                
                ens_dist[i] = res.fun
                r.reset()
                ens_model[i] = r.getAntimony(current=True)
                ens_stoi[i] = st
                ens_rtypes[i] = rTypes
                ens_ia[i] = ia
                ens_concCC[i] = concCC
                
        if Settings.checkCorrectStoichiometry and Settings.MODEL_INPUT == None:
            ens_stt = copy.deepcopy(ens_stoi[:,realFloatingIdsInd])
            ens_stt[ens_stt > 0] = 1
            ens_stt[ens_stt < 0] = -1
            if np.any(np.all(realStoi == ens_stt, axis=(1, 2))):
                print('The algorithm recovered the original model')
            else:
                print('The original model is not in the population')


        if Settings.refine:
            print("Refining")
            for i,j in enumerate(ens_model):
                try:
                    r = te.loada(j)
                    gpid = [x for x in r.getGlobalParameterIds() if not x.startswith('S')]
                    p_bound = ng.generateParameterBoundary(gpid)
                    res = scipy.optimize.differential_evolution(f1, args=(r,), 
                                        bounds=p_bound, maxiter=Settings.optiMaxIter, 
                                        tol=Settings.refineTol, polish=True, 
                                        seed=Settings.r_seed)
                    
                    if res.success and res.fun < 1e6:
                        r.resetToOrigin()
                        r.setValues(gpid, res.x)
                        ens_dist[i] = res.fun
                        ens_model[i] = r.getAntimony(current=True)
                except:
                    pass
                try:
                    r.clearModel()
                except:
                    pass        
                antimony.clearPreviousLoads()
                antimony.freeAll()
            
            best_dist[-1] = dist_top[0]
            avg_dist[-1] = np.average(dist_top)
            med_dist[-1] = np.median(dist_top)
            top_dist[-1] = np.average(np.unique(dist_top)[:top_ind])
        
        
        # Replace correct species names
        for i,j in enumerate(ens_stoi):
            r = te.loada(ens_model[i])
            param = r.getGlobalParameterValues()
            newAnt = ng.generateAntimonyfromST(list(fid_dict.values()), list(bid_dict.values()),
                                               j, ens_rtypes[i], ens_ia[i],
                                               boundary_init=realBoundaryVal)
            # TODO: remove
            r = te.loada(newAnt)
            r.setValues(r.getGlobalParameterIds(), param)
            ens_model[i] = r.getAntimony(current=True)
            try:
                r.clearModel()
            except:
                pass        
            antimony.clearPreviousLoads()
            antimony.freeAll()

        memory.append(process.memory_info().rss)
        dist_top_ind = np.argsort(ens_dist)
        dist_top = ens_dist[dist_top_ind]
        model_top = ens_model[dist_top_ind]

        # Collect models
        kdeOutput = analysis.selectWithKernalDensity(dist_top)
        
        if Settings.EXPORT_ALL_MODELS:
            model_col = model_top
            dist_col = dist_top
            stoi_col = ens_stoi[dist_top_ind]
            rtypes_col = ens_rtypes[dist_top_ind]
            ia_col = ens_ia[dist_top_ind]
            concCC_col = ens_concCC[dist_top_ind]
        else:
            model_col = model_top[:kdeOutput[0]]
            dist_col = dist_top[:kdeOutput[0]]
            stoi_col = ens_stoi[dist_top_ind][:kdeOutput[0]]
            rtypes_col = ens_rtypes[dist_top_ind][:kdeOutput[0]]
            ia_col = ens_ia[dist_top_ind][:kdeOutput[0]]
            concCC_col = ens_concCC[dist_top_ind][:kdeOutput[0]]
            
#%%
        Settings.EXPORT_PATH = ioutils.exportPathHandler(Settings)

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
        if Settings.EXPORT_SETTINGS:
            ioutils.exportSettings(Settings, path=Settings.EXPORT_PATH)
        
        if Settings.EXPORT_OUTPUT:
            ioutils.exportOutputs(dist_col, [best_dist, avg_dist, med_dist, top_dist], 
                                  Settings, tracking, path=Settings.EXPORT_PATH)
            ioutils.exportModels(model_col, path=Settings.EXPORT_PATH)
            ioutils.exportReport(model_col, Settings, runtime, tracking, len(top_dist), 
                                 path=Settings.EXPORT_PATH)
        
        if Settings.EXPORT_CACHE:
            ioutils.exportModelComponents(stoi_col, rtypes_col, ia_col,
                                          path=Settings.EXPORT_PATH)
            ioutils.exportControlCoefficients(concCC_col, path=Settings.EXPORT_PATH)

    
    # Restore config
    roadrunner.Config.setValue(roadrunner.Config.PYTHON_ENABLE_NAMED_MATRIX, rr_nmval)

