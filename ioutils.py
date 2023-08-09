# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:18:02 2018

@author: ckiri
"""

import os, shutil
import warnings
import numpy as np
import pandas as pd
import analysis


def exportPathHandler(Settings):
    """
    Generate export path
    
    :param Settings: Settings class
    :return path: absolute export path
    """
    
    if Settings.EXPORT_PATH != None:
        if Settings.EXPORT_FORCE_MODELNAMES:
            if Settings.MODEL_INPUT != None:
                fname = os.path.basename(Settings.MODEL_INPUT).split('.')[0]
            else:
                fname = Settings.modelType
            path = os.path.join(Settings.EXPORT_PATH, fname)
            path = os.path.abspath(os.path.join(os.getcwd(), path))
        else:
            path = os.path.abspath(os.path.join(os.getcwd(), Settings.EXPORT_PATH))
    else:
        if Settings.MODEL_INPUT != None:
            fname = os.path.basename(Settings.MODEL_INPUT).split('.')[0]
            path = os.path.abspath(os.path.join(os.getcwd(), fname))
        else:
            path = os.path.abspath(os.path.join(os.getcwd(), 'evoMEG_output'))
    
    
    if (Settings.SAVE_PLOT or Settings.EXPORT_SETTINGS or Settings.EXPORT_OUTPUT 
        or Settings.EXPORT_ALL_MODELS):
        try:
            if Settings.EXPORT_OVERWRITE:
                if not os.path.exists(path):
                    os.makedirs(path)
                else:
                    shutil.rmtree(path)
                    os.makedirs(path)
            else:
                if not os.path.exists(path):
                    os.makedirs(path)
                else:
                    f, ext = os.path.splitext(path)
                    counter = 0
                    while os.path.exists(path):
                        path = f + "_" + str(counter) + ext
                        counter += 1
                    os.makedirs(path)
        except:
            raise Exception('Failed to create the output directory')
            
    return path


def validateSettings(Settings):
    """
    Validate the Settings parameters
    
    :param Settings: Settings class
    """
    
    assert(isinstance(Settings.MODEL_INPUT, str) or Settings.MODEL_INPUT == None), 'Invalid MODEL_INPUT'
    assert(isinstance(Settings.DATA_INPUT, str) or Settings.DATA_INPUT == None), 'Invalid DATA_INPUT'
    assert(isinstance(Settings.READ_SETTINGS, str) or Settings.READ_SETTINGS == None), 'Invalid READ_SETTINGS'
    assert(isinstance(Settings.CACHED_DIR, str) or Settings.CACHED_DIR == None), 'Invalid CACHED_DIR'
    
    assert(isinstance(Settings.ens_size, int)), 'Invalid ens_size'
    assert(isinstance(Settings.pass_size, int)), 'Invalid pass_size'
    assert(isinstance(Settings.top_p, float) and Settings.top_p<=1), 'Invalid top_p'
    assert(isinstance(Settings.maxIter_init, int)), 'Invalid maxIter_init'
    assert(isinstance(Settings.maxIter_gen, int)), 'Invalid maxIter_gen'
    assert(isinstance(Settings.maxIter_mut, int)), 'Invalid maxIter_mut'
    assert(isinstance(Settings.recomb, float) and Settings.recomb<=1), 'Invalid recomb'
    assert(isinstance(Settings.conservedMoiety, bool)), 'Invalid conservedMoiety'
    assert(isinstance(Settings.checkCorrectStoichiometry, bool)), 'Invalid checkCorrectStoichiometry'
    
    assert(isinstance(Settings.n_gen, int) or Settings.n_gen == None), 'Invalid n_gen'
    assert(isinstance(Settings.gen_static, int) or Settings.gen_static == None), 'Invalid gen_static'
    assert(isinstance(Settings.thres_avg, (int, float)) or Settings.thres_avg == None), 'Invalid thres_avg'
    assert(isinstance(Settings.thres_median, (int, float)) or Settings.thres_median == None), 'Invalid thres_median'
    assert(isinstance(Settings.thres_shortest, (int, float)) or Settings.thres_shortest == None), 'Invalid thres_shortest'
    assert(isinstance(Settings.thres_top, (int, float)) or Settings.thres_top == None), 'Invalid thres_top'
    assert(isinstance(Settings.max_run_time, (int, float)) or Settings.max_run_time == None), 'Invalid max_run_time'
    assert([Settings.n_gen, Settings.gen_static, Settings.thres_avg, Settings.thres_median,
            Settings.thres_shortest, Settings.thres_top] != [None, None, None, None, None, None]), 'No termination condition given'
    
    assert(isinstance(Settings.optiMaxIter, int)), 'Invalid optiMaxIter'
    assert(isinstance(Settings.optiTol, (int, float))), 'Invalid optiTol'
    assert(isinstance(Settings.optiPolish, bool)), 'Invalid optiPolish'
    assert(isinstance(Settings.refine, bool)), 'Invalid refine'
    assert(isinstance(Settings.refineTol, (int, float))), 'Invalid refineTol'
    
    assert(isinstance(Settings.r_seed, int)), 'Invalid r_seed'
    assert(isinstance(Settings.NOISE, bool)), 'Invalid NOISE'
    assert(isinstance(Settings.ABS_NOISE_STD, (int, float))), 'Invalid ABS_NOISE_STD'
    assert(isinstance(Settings.REL_NOISE_STD, (int, float))), 'Invalid REL_NOISE_STD'
    
    assert(isinstance(Settings.SHOW_PLOT, bool)), 'Invalid SHOW_PLOT'
    assert(isinstance(Settings.SAVE_PLOT, bool)), 'Invalid SAVE_PLOT'
    assert(isinstance(Settings.EXPORT_ALL_MODELS, bool)), 'Invalid EXPORT_ALL_MODELS'
    assert(isinstance(Settings.EXPORT_OUTPUT, bool)), 'Invalid EXPORT_OUTPUT'
    assert(isinstance(Settings.EXPORT_SETTINGS, bool)), 'Invalid EXPORT_SETTINGS'
    assert(isinstance(Settings.EXPORT_CACHE, bool)), 'Invalid EXPORT_CACHE'
    assert(isinstance(Settings.EXPORT_PATH, str) or Settings.EXPORT_PATH == None), 'Invalid EXPORT_PATH'
    assert(isinstance(Settings.EXPORT_OVERWRITE, bool)), 'Invalid EXPORT_OVERWRITE'
    assert(isinstance(Settings.EXPORT_FORCE_MODELNAMES, bool)), 'Invalid EXPORT_FORCE_MODELNAMES'
    
    
def exportSettings(Settings, path=None):
    """
    Export all settings to a specified path
    
    :param Settings: Settings class
    :param path: path to export settings
    """
    
    if path:
        outputdir = path
    else:
        outputdir = os.path.join(os.getcwd(), 'output')
        
    outputtxt = open(os.path.join(outputdir, 'settings.txt'), 'w')
    if Settings.MODEL_INPUT != None:
        outputtxt.writelines('MODEL_INPUT: {}'.format(Settings.MODEL_INPUT) + '\n')
    else:
        outputtxt.writelines('modelType: {}'.format(Settings.modelType) + '\n')
    if Settings.DATA_INPUT != None:
        outputtxt.writelines('DATA_INPUT: {}'.format(Settings.DATA_INPUT) + '\n')
    if Settings.CACHED_DIR != None:
        outputtxt.writelines('CACHED_DIR: {}'.format(Settings.CACHED_DIR) + '\n')
    outputtxt.writelines('ens_size: {}'.format(Settings.ens_size) + '\n')
    outputtxt.writelines('pass_size: {}'.format(Settings.pass_size) + '\n')
    outputtxt.writelines('top_p: {}'.format(Settings.top_p) + '\n')
    outputtxt.writelines('maxIter_init: {}'.format(Settings.maxIter_init) + '\n')
    outputtxt.writelines('maxIter_gen: {}'.format(Settings.maxIter_gen) + '\n')
    outputtxt.writelines('maxIter_mut: {}'.format(Settings.maxIter_mut) + '\n')
    outputtxt.writelines('recomb: {}'.format(Settings.recomb) + '\n')
    outputtxt.writelines('n_gen: {}'.format(Settings.n_gen) + '\n')
    outputtxt.writelines('gen_static: {}'.format(Settings.gen_static) + '\n')
    outputtxt.writelines('thres_avg: {}'.format(Settings.thres_avg) + '\n')
    outputtxt.writelines('thres_median: {}'.format(Settings.thres_median) + '\n')
    outputtxt.writelines('thres_shortest: {}'.format(Settings.thres_shortest) + '\n')
    outputtxt.writelines('thres_top: {}'.format(Settings.thres_top) + '\n')
    outputtxt.writelines('max_run_time: {}'.format(Settings.max_run_time) + '\n')
    outputtxt.writelines('optiMaxIter: {}'.format(Settings.optiMaxIter) + '\n')
    outputtxt.writelines('optiTol: {}'.format(Settings.optiTol) + '\n')
    outputtxt.writelines('optiPolish: {}'.format(Settings.optiPolish) + '\n')
    outputtxt.writelines('refine: {}'.format(Settings.refine) + '\n')
    outputtxt.writelines('refineTol: {}'.format(Settings.refineTol) + '\n')
    outputtxt.writelines('r_seed: {}'.format(Settings.r_seed) + '\n')
    outputtxt.writelines('NOISE: {}'.format(Settings.NOISE) + '\n')
    outputtxt.writelines('ABS_NOISE_STD: {}'.format(Settings.ABS_NOISE_STD) + '\n')
    outputtxt.writelines('REL_NOISE_STD: {}'.format(Settings.REL_NOISE_STD) + '\n')
    outputtxt.close()
    
    
def exportOutputs(dists, dist_list, Settings, tracking, path=None):
    """
    Export all outputs to a specified path
        
    :param path: path to export outputs
    """
    
    if path:
        outputdir = path
    else:
        outputdir = os.path.join(os.getcwd(), 'output')
        
    df = pd.DataFrame(np.array(dists), columns=['distance'])
    df.to_csv(os.path.join(outputdir, 'dist_collected.txt'))
    
    stat = pd.DataFrame(np.array(dist_list).T, 
                        columns=['generation best', 
                                 'generation average',
                                 'generation median',
                                 'generation top {}'.format(int(Settings.top_p*100))])
    stat.to_csv(os.path.join(outputdir, 'dist_stat.txt'))
    
    tracking_arr = np.array(tracking)
    np.save(os.path.join(outputdir, 'tracking.npy'), tracking_arr, allow_pickle=True)
    

def exportModels(models, path=None):
    """
    Export models to a specified path
        
    :param path: path to export the models
    """
    
    if path:
        outputdir = path
    else:
        outputdir = os.path.join(os.getcwd(), 'output')
        
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    if not os.path.exists(os.path.join(outputdir, 'models')):
        os.mkdir(os.path.join(outputdir, 'models'))
        
    for i in range(len(models)):
        modeltxt = open(os.path.join(outputdir, 'models/model_%03d' % i + '.txt'), 'w')
        modeltxt.write(models[i])
        modeltxt.close()
    
    
def exportModelComponents(stoi, rtypes, ia, path=None):
    """
    Export model components to a specified path
        
    :param path: path to export the model components
    """
    
    if path:
        outputdir = path
    else:
        outputdir = os.path.join(os.getcwd(), 'output')
        
    np.save(os.path.join(outputdir, 'stoi.npy'), stoi, allow_pickle=True)
    np.save(os.path.join(outputdir, 'rtypes.npy'), rtypes, allow_pickle=True)
    np.save(os.path.join(outputdir, 'ia.npy'), ia, allow_pickle=True)
    
    
def exportReport(models, Settings, time, tracking, n, path=None):
    """
    Export the report file to a specified path
        
    :param path: path to export the report file
    """
    
    if path:
        outputdir = path
    else:
        outputdir = os.path.join(os.getcwd(), 'output')
    
    outputtxt = open(os.path.join(outputdir, 'report.txt'), 'w')
    outputtxt.writelines('------------------------- REPORT -------------------------\n')
    outputtxt.writelines('RUN COMPLETE. HERE ARE SOME METRIC YOU MIGHT BE INTERESTED\n')
    outputtxt.writelines('No. of Generations: {}'.format(n) + '\n')
    outputtxt.writelines('Ensemble Size: {}'.format(Settings.ens_size) + '\n')
    outputtxt.writelines('No. of Collected Models: {}'.format(len(models)) + '\n')
    outputtxt.writelines('Run Time: {:.2f}'.format(time) + ' s\n')
    outputtxt.writelines('No. Stoich. Analyzed: {}'.format(len(tracking)) + '\n')
    outputtxt.close()


def exportControlCoefficients(concCC, path=None):
    """
    Export model components to a specified path
        
    :param path: path to export the model components
    """
    
    if path:
        outputdir = path
    else:
        outputdir = os.path.join(os.getcwd(), 'output')
        
    np.save(os.path.join(outputdir, 'concCC.npy'), concCC, allow_pickle=True)
    

def readSettings(Settings):
    """
    Read setting file and update Settings class
    
    :param Settings: Settings class
    """
    
    if os.path.isfile(Settings.READ_SETTINGS):
        f = open(Settings.READ_SETTINGS, "r")
        ls = f.read().splitlines()
        f.close()
        
        for s in ls:
            sp = s.split(': ')
            try:
                if sp[1].replace('.','',1).isdigit():
                    if sp[1].isdigit():
                        Settings.__setattr__(sp[0], int(sp[1]))
                    else:
                        Settings.__setattr__(sp[0], float(sp[1]))
                elif sp[1] == 'True':
                    Settings.__setattr__(sp[0], True)
                elif sp[1] == 'False':
                    Settings.__setattr__(sp[0], False)
                elif sp[1] == 'None':
                    Settings.__setattr__(sp[0], None)
                else:
                    if isinstance(sp[1], str):
                        s = sp[1].strip("'")
                        s = s.strip('"')
                        Settings.__setattr__(sp[0], s)
                    else:
                        Settings.__setattr__(sp[0], sp[1])
            except:
                warnings.warn("Setting {} is not valid and was ignored".format(s))
    else:
        raise warnings.warn("Cannot read the settings file")
    

def readModels(modelsPath):
    """
    Read model files
    
    :param modelsPath: path to a directory containing model files
    :returns: list of model strings
    """
    
    modelfiles = [f for f in os.listdir(modelsPath) if os.path.isfile(os.path.join(modelsPath, f))]

    antstr = []
    for i in modelfiles:
        sbmlstr = open(os.path.join(modelsPath, i), 'r')
        antstr.append(sbmlstr.read())
        sbmlstr.close()
    
    return antstr


def readData(dataPath):
    """
    Read data encoded in csv.
    
    :param dataPath: path to a csv file
    :returns: DataFrame
    """
    
    if os.path.exists(dataPath):
        df = pd.read_csv(dataPath)
        return df
    else:
        raise Exception("Cannot find the file at the specified path")


def readCache(Settings):
    """
    Read cached output.
    
    :param dataPath: path to the cached output directory
    """
    
    try:
        ens_model = np.empty(Settings.ens_size, dtype='object')
        for i in range(Settings.ens_size):
            modeltxt = open(os.path.join(Settings.CACHED_DIR, 'models/model_%03d' % i + '.txt'), 'r')
            ens_model[i] = modeltxt.read()
            modeltxt.close()
    except:
        raise Exception("Cannot read models")
    
    dist_path = os.path.join(Settings.CACHED_DIR, 'dist_collected.txt')
    concCC_path = os.path.join(Settings.CACHED_DIR, 'concCC.npy')
    if os.path.isfile(dist_path) and os.path.isfile(concCC_path):
        ens_dist = pd.read_csv(dist_path, index_col=0)
        ens_dist = ens_dist.to_numpy().flatten()
        ens_concCC = np.load(concCC_path, allow_pickle=True)
    else:
        ens_dist, ens_concCC = analysis.cacheFallBack1(Settings, ens_model)
    
    stoi_path = os.path.join(Settings.CACHED_DIR, 'stoi.npy')
    rtypes_path = os.path.join(Settings.CACHED_DIR, 'rtypes.npy')
    ia_path = os.path.join(Settings.CACHED_DIR, 'ia.npy')
    tracking_path = os.path.join(Settings.CACHED_DIR, 'tracking.npy')
    
    if (os.path.isfile(stoi_path) and os.path.isfile(rtypes_path) and 
        os.path.isfile(ia_path) and os.path.isfile(tracking_path)):
        ens_stoi = np.load(stoi_path, allow_pickle=True)
        ens_rtypes = np.load(rtypes_path, allow_pickle=True)
        ens_ia = np.load(ia_path, allow_pickle=True)
        tracking = np.load(tracking_path, allow_pickle=True).tolist()
    else:
        (ens_stoi, ens_rtypes, ens_ia, tracking) = analysis.cacheFallBack2(Settings, ens_model)
    
    return (ens_dist, ens_model, ens_stoi, ens_rtypes, ens_ia, ens_concCC, tracking)
    
    
    
def readStats(Settings):
    """
    Read cached output.
    
    :param dataPath: path to the cached output directory
    """
    try:
        stats = pd.read_csv(os.path.join(Settings.CACHED_DIR, 'dist_stat.txt'), index_col=0)
        
        best_dist = stats['generation best'].tolist()[:-1]
        avg_dist = stats['generation average'].tolist()[:-1]
        med_dist = stats['generation median'].tolist()[:-1]
        top_dist = stats['generation top {}'.format(int(Settings.top_p*100))].tolist()[:-1]
    except:
        warnings.warn("Cannot read the fitness statistics")
        best_dist = []
        avg_dist = []
        med_dist = []
        top_dist = []
    
    return (best_dist, avg_dist, med_dist, top_dist)
    
    
def testModels(modelType):
    """
    Returns a test model
    
    :param modelType: model name, e.g. 'FFL', 'Linear', 'Nested', 'Branched'
    :returns: Antimony string
    """
    
    if modelType == 'Linear_m':
        # Linear    
        realModel = """
        var S1, S2, S3, S4;
        const S0, S5;
        J0: S0 -> S1; Kf0*S0/(1 + S0);
        J1: S1 -> S2; Kf1*S1/(1 + S1);
        J2: S2 -> S3; Kf2*S2/(1 + S2);
        J3: S3 -> S4; Kf3*S3/(1 + S3);
        J4: S4 -> S5; Kf4*S4/(1 + S4);
       
        Kf0 = 0.285822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.27503984992
       
        S0 = 3
        S5 = 5
        S1 = 1
        S2 = 1
        S3 = 1
        S4 = 1
        """
    elif modelType == 'Linear_r':
        # Linear    
        realModel = """
        var S1, S2, S3, S4;
        const S0, S5;
        J0: S0 -> S1; (Kf0*S0 - Kr0*S1)/(1 + S0 + S1);
        J1: S1 -> S2; (Kf1*S1 - Kr1*S2)/(1 + S1 + S2);
        J2: S2 -> S3; (Kf2*S2 - Kr2*S3)/(1 + S2 + S3);
        J3: S3 -> S4; (Kf3*S3 - Kr3*S4)/(1 + S3 + S4);
        J4: S4 -> S5; (Kf4*S4 - Kr4*S5)/(1 + S4 + S5);
       
        Kf0 = 0.285822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.27503984992
        Kr0 = 0.072790656829
        Kr1 = 0.27828563882
        Kr2 = 0.166906190765
        Kr3 = 0.27057924345
        Kr4 = 0.1026847763
       
        S0 = 3
        S5 = 5
        S1 = 1
        S2 = 1
        S3 = 1
        S4 = 1
        """
    elif modelType == 'Nested_m':
        # Nested
        realModel = """
        var S1, S2, S3;
        const S0, S4;
        J0: S0 -> S1; Kf0*S0/(1 + S0);
        J1: S1 -> S2; Kf1*S1/(1 + S1);
        J2: S2 -> S1; Kf2*S2/(1 + S2);
        J3: S1 -> S3; Kf3*S1/(1 + S1);
        J4: S3 -> S1; Kf4*S3/(1 + S3);
        J5: S3 -> S4; Kf5*S3/(1 + S3);
       
        Kf0 = 0.285822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.148522702962
        Kf5 = 0.348927696783
       
        S0 = 3
        S4 = 5
        S1 = 1
        S2 = 1
        S3 = 1
        """
    elif modelType == 'Nested_r':
        # Nested
        realModel = """
        var S1, S2, S3;
        const S0, S4;
        J0: S0 -> S1; (Kf0*S0 - Kr0*S1)/(1 + S0 + S1);
        J1: S1 -> S2; (Kf1*S1 - Kr1*S2)/(1 + S1 + S2);
        J2: S2 -> S1; (Kf2*S2 - Kr2*S1)/(1 + S2 + S1);
        J3: S1 -> S3; (Kf3*S1 - Kr3*S3)/(1 + S1 + S3);
        J4: S3 -> S1; (Kf4*S3 - Kr4*S1)/(1 + S3 + S1);
        J5: S3 -> S4; (Kf5*S3 - Kr5*S4)/(1 + S3 + S4);
       
        Kf0 = 0.285822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.148522702962
        Kf5 = 0.348927696783
        Kr0 = 0.072790656829
        Kr1 = 0.27828563882
        Kr2 = 0.166906190765
        Kr3 = 0.27057924345
        Kr4 = 0.1026847763
        Kr5 = 0.185479288476
       
        S0 = 3
        S4 = 5
        S1 = 1
        S2 = 1
        S3 = 1
        """
    elif modelType == 'Feedback_m':
        # FFL    
        realModel = """
        var S1, S2, S3;
        const S0, S4;
        J0: S0 -> S1; Kf0*S0/(1 + S0);
        J1: S1 -> S2; Kf1*S1/(1 + S1);
        J2: S2 -> S3; Kf2*S2/(1 + S2);
        J3: S3 -> S4; Kf3*S3/(1 + S3);
        J4: S3 -> S1; Kf4*S3/(1 + S3);
       
        Kf0 = 0.285822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.148522702962
       
        S0 = 3
        S4 = 5
        S1 = 1
        S2 = 1
        S3 = 1
        """
    elif modelType == 'Feedback_r':
        # FFL    
        realModel = """
        var S1, S2, S3;
        const S0, S4;
        J0: S0 -> S1; (Kf0*S0 - Kr0*S1)/(1 + S0 + S1);
        J1: S1 -> S2; (Kf0*S1 - Kr1*S2)/(1 + S1 + S2);
        J2: S2 -> S3; (Kf2*S2 - Kr2*S3)/(1 + S2 + S3);
        J3: S3 -> S4; (Kf3*S3 - Kr3*S4)/(1 + S3 + S4);
        J4: S3 -> S1; (Kf4*S3 - Kr4*S1)/(1 + S3 + S1);
       
        Kf0 = 0.285822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.548522702962
        Kr0 = 0.072790656829
        Kr1 = 0.27828563882
        Kr2 = 0.166906190765
        Kr3 = 0.27057924345
        Kr4 = 0.1026847763
       
        S0 = 3
        S4 = 5
        S1 = 1
        S2 = 1
        S3 = 1
        """
    elif modelType == 'FFL_m':
        # FFL    
        realModel = """
        var S1, S2, S3;
        const S0, S4;
        J0: S0 -> S1; Kf0*S0/(1 + S0);
        J1: S1 -> S2; Kf1*S1/(1 + S1);
        J2: S2 -> S3; Kf2*S2/(1 + S2);
        J3: S3 -> S4; Kf3*S3/(1 + S3);
        J4: S1 -> S3; Kf4*S1/(1 + S1);
       
        Kf0 = 0.285822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.148522702962
       
        S0 = 3
        S4 = 5
        S1 = 1
        S2 = 1
        S3 = 1
        """
    elif modelType == 'FFL_r':
        # FFL    
        realModel = """
        var S1, S2, S3;
        const S0, S4;
        J0: S0 -> S1; (Kf0*S0 - Kr0*S1)/(1 + S0 + S1);
        J1: S1 -> S2; (Kf0*S1 - Kr1*S2)/(1 + S1 + S2);
        J2: S2 -> S3; (Kf2*S2 - Kr2*S3)/(1 + S2 + S3);
        J3: S3 -> S4; (Kf3*S3 - Kr3*S4)/(1 + S3 + S4);
        J4: S1 -> S3; (Kf4*S1 - Kr4*S3)/(1 + S1 + S3);
       
        Kf0 = 0.285822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.548522702962
        Kr0 = 0.072790656829
        Kr1 = 0.27828563882
        Kr2 = 0.166906190765
        Kr3 = 0.27057924345
        Kr4 = 0.1026847763
       
        S0 = 3
        S4 = 5
        S1 = 1
        S2 = 1
        S3 = 1
        """
    elif modelType == 'FFL_m_a':
        # FFL    
        realModel = """
        var S1, S2, S3;
        const S0, S4;
        J0: S0 -> S1; Kf0*S0/(1 + S0);
        J1: S1 -> S2; Kf1*S1/(1 + S1);
        J2: S2 -> S3; Kf2*S2/(1 + S2);
        J3: S3 -> S4; Kf3*S3/(1 + S3);
        J4: S1 -> S3; (Kf4*S1/(1 + S1))*(1 + Ka0*S2);
       
        Kf0 = 0.285822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.148522702962
        Ka0 = 0.883848629231
       
        S0 = 3
        S4 = 5
        S1 = 1
        S2 = 1
        S3 = 1
        """
    elif modelType == 'FFL_m_i':
        # I1FFL
        realModel = """
        var S1, S2, S3;
        const S0, S4;
        J0: S0 -> S1; Kf0*S0/(1 + S0);
        J1: S1 -> S2; Kf1*S1/(1 + S1);
        J2: S2 -> S3; Kf2*S2/(1 + S2);
        J3: S3 -> S4; Kf3*S3/(1 + S3);
        J4: S1 -> S3; (Kf4*S1/(1 + S1))*1/(1 + Ki0*S2);
       
        Kf0 = 0.285822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.148522702962
        Ki0 = 0.974569278466
       
        S0 = 3
        S4 = 5
        S1 = 1
        S2 = 1
        S3 = 1
        """
    elif modelType == 'FFL_r_a':
        # FFL    
        realModel = """
        var S1, S2, S3;
        const S0, S4;
        J0: S0 -> S1; (Kf0*S0 - Kr0*S1)/(1 + S0 + S1);
        J1: S1 -> S2; (Kf0*S1 - Kr1*S2)/(1 + S1 + S2);
        J2: S2 -> S3; (Kf2*S2 - Kr2*S3)/(1 + S2 + S3);
        J3: S3 -> S4; (Kf3*S3 - Kr3*S4)/(1 + S3 + S4);
        J4: S1 -> S3; (1 + Ka0*S2)*(Kf4*S1 - Kr4*S3)/(1 + S1 + S3);
       
        Kf0 = 0.285822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.548522702962
        Kr0 = 0.072790656829
        Kr1 = 0.27828563882
        Kr2 = 0.166906190765
        Kr3 = 0.27057924345
        Kr4 = 0.1026847763
        Ka0 = 0.883848629231
       
        S0 = 3
        S4 = 5
        S1 = 1
        S2 = 1
        S3 = 1
        """
    elif modelType == 'FFL_r_i':
        # FFL    
        realModel = """
        var S1, S2, S3;
        const S0, S4;
        J0: S0 -> S1; (Kf0*S0 - Kr0*S1)/(1 + S0 + S1);
        J1: S1 -> S2; (Kf0*S1 - Kr1*S2)/(1 + S1 + S2);
        J2: S2 -> S3; (Kf2*S2 - Kr2*S3)/(1 + S2 + S3);
        J3: S3 -> S4; (Kf3*S3 - Kr3*S4)/(1 + S3 + S4);
        J4: S1 -> S3; 1/(1 + Ki0*S2)*(Kf4*S1 - Kr4*S3)/(1 + S1 + S3);
       
        Kf0 = 0.285822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.548522702962
        Kr0 = 0.072790656829
        Kr1 = 0.27828563882
        Kr2 = 0.166906190765
        Kr3 = 0.27057924345
        Kr4 = 0.1026847763
        Ki0 = 0.974569278466
       
        S0 = 3
        S4 = 5
        S1 = 1
        S2 = 1
        S3 = 1
        """
    elif modelType == 'Branched_m':
        #Branched
        realModel = """
        var S1, S2, S3, S4, S5;
        const S0, S6;
        J0: S0 -> S1; Kf0*S0/(1 + S0);
        J1: S1 -> S2; Kf1*S1/(1 + S1);
        J2: S1 -> S3; Kf2*S1/(1 + S1);
        J3: S3 -> S4; Kf3*S3/(1 + S3);
        J4: S3 -> S5; Kf4*S3/(1 + S3);
        J5: S2 -> S6; Kf5*S2/(1 + S2);
        J6: S4 -> S6; Kf6*S4/(1 + S4);
        J7: S5 -> S6; Kf7*S5/(1 + S5);
       
        Kf0 = 0.285822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.148522702962
        Kf5 = 0.348927696783
        Kf6 = 0.572677236248
        Kf7 = 0.497208763889
       
        S0 = 3
        S6 = 5
        S1 = 1
        S2 = 1
        S3 = 1
        S4 = 1
        S5 = 1
        """
    elif modelType == 'Branched_r':
        #Branched
        realModel = """
        var S1, S2, S3, S4, S5;
        const S0, S6;
        J0: S0 -> S1; (Kf0*S0 - Kr0*S1)/(1 + S0 + S1);
        J1: S1 -> S2; (Kf1*S1 - Kr1*S2)/(1 + S1 + S2);
        J2: S1 -> S3; (Kf2*S1 - Kr2*S3)/(1 + S1 + S3);
        J3: S3 -> S4; (Kf3*S3 - Kr3*S4)/(1 + S3 + S4);
        J4: S3 -> S5; (Kf4*S3 - Kr4*S5)/(1 + S3 + S5);
        J5: S2 -> S6; (Kf5*S2 - Kr5*S6)/(1 + S2 + S6);
        J6: S4 -> S6; (Kf6*S4 - Kr6*S6)/(1 + S4 + S6);
        J7: S5 -> S6; (Kf7*S5 - Kr7*S6)/(1 + S5 + S6);
       
        Kf0 = 0.285822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.148522702962
        Kf5 = 0.348927696783
        Kf6 = 0.572677236248
        Kf7 = 0.497208763889
        Kr0 = 0.072790656829
        Kr1 = 0.27828563882
        Kr2 = 0.166906190765
        Kr3 = 0.27057924345
        Kr4 = 0.1026847763
        Kr5 = 0.185479288476
        Kr6 = 0.34908380750
        Kr7 = 0.16784787349
       
        S0 = 3
        S6 = 5
        S1 = 1
        S2 = 1
        S3 = 1
        S4 = 1
        S5 = 1
        """
    elif modelType == 'sigPath':
        realModel = """
        var S0, S1, S2, S3, S4, S5;
        J0: S0 -> S1; 1/(1 + S5/Ki0)*(Kf0*S0)/(1 + S0);
        J1: S1 -> S0; (Kf1*S1)/(1 + S1);
        J2: S2 -> S3; (1 + Ka2*S1)*(Kf2*S2)/(1 + S2);
        J3: S3 -> S2; (Kf3*S3)/(1 + S3);
        J4: S4 -> S5; (1 + Ka4*S3)*(Kf4*S4)/(1 + S4);
        J5: S5 -> S4; (Kf5*S5)/(1 + S5);
        
        Kf0 = 0.485822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.948522702962
        Kf5 = 0.272790656829
        Ki0 = 0.34569278466
        Ka2 = 0.6276983967
        Ka4 = 0.1143526464
        
        S0 = 1
        S1 = 1
        S2 = 1
        S3 = 1
        S4 = 1
        S5 = 1
        """
    elif modelType == 'Full2':
        realModel = """
        var S0, S1, S2, S3, S4, S5;
        const S6;
        J0: S0 -> S1; 1/(1 + S5/Ki0)*(Kf0*S0)/(1 + S0);
        J1: S1 -> S0; (Kf1*S1)/(1 + S1);
        J2: S2 -> S3; (1 + Ka2*S1)*(Kf2*S2)/(1 + S2);
        J3: S3 -> S2; (Kf3*S3)/(1 + S3);
        J4: S4 -> S5; (1 + Ka4*S3)*(Kf4*S4)/(1 + S4);
        J5: S5 -> S4; (Kf5*S5)/(1 + S5);
        J6: S5 -> S6; (Kf6*S5)/(1 + S5);
        
        Kf0 = 0.485822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.948522702962
        Kf5 = 0.272790656829
        Kf6 = 0.7285073956
        Ki0 = 0.34569278466
        Ka2 = 0.6276983967
        Ka4 = 0.1143526464
        
        S6 = 3
        S0 = 1
        S1 = 1
        S2 = 1
        S3 = 1
        S4 = 1
        S5 = 1
        """
    elif modelType == 'Linear_r_a':
        realModel = """
        var S1, S2, S3;
        const S0, S4;
        J0: S0 -> S1; (Kf0*S0 - Kr0*S1)/(1 + S0 + S1);
        J1: S1 -> S2; (Kf1*S1 - Kr1*S2)/(1 + S1 + S2);
        J2: S2 -> S3; (Kf2*S2 - Kr2*S3)/(1 + S2 + S3);
        J3: S3 -> S4; (Kf3*S3 - Kr3*S4)/(1 + S3 + S4)*(S1/(1 + Ka0*S1));
       
        Kf0 = 0.885822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kr0 = 0.072790656829
        Kr1 = 0.27828563882
        Kr2 = 0.166906190765
        Kr3 = 0.31057924345
        Ka0 = 0.723848629231
       
        S0 = 3
        S1 = 1
        S2 = 1
        S3 = 1
        S4 = 1
        """
    elif modelType == 'Linear_r_i':
        realModel = """
        var S1, S2, S3;
        const S0, S4;
        J0: S0 -> S1; (Kf0*S0 - Kr0*S1)/(1 + S0 + S1);
        J1: S1 -> S2; (Kf1*S1 - Kr1*S2)/(1 + S1 + S2);
        J2: S2 -> S3; (Kf2*S2 - Kr2*S3)/(1 + S2 + S3);
        J3: S3 -> S4; (Kf3*S3 - Kr3*S4)/(1 + S3 + S4)*(1/(1 + Ki0*S1));
       
        Kf0 = 0.885822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kr0 = 0.072790656829
        Kr1 = 0.27828563882
        Kr2 = 0.166906190765
        Kr3 = 0.31057924345
        Ki0 = 0.814569278466
       
        S0 = 3
        S1 = 1
        S2 = 1
        S3 = 1
        S4 = 1
        """
    elif modelType == 'Linear_r_a_allo':
        realModel = """
        var S1, S2, S3;
        const S0, S4;
        J0: S0 -> S1; (Kf0*S0 - Kr0*S1)/(1 + S0 + S1);
        J1: S1 -> S2; (Kf1*S1 - Kr1*S2)/(1 + S1 + S2);
        J2: S2 -> S3; (Kf2*S2 - Kr2*S3)/(1 + S2 + S3);
        J3: S3 -> S4; (Kf3*S3 - Kr3*S4)/(1 + S3 + S4 + Ka0/S1);
       
        Kf0 = 0.885822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kr0 = 0.072790656829
        Kr1 = 0.27828563882
        Kr2 = 0.166906190765
        Kr3 = 0.31057924345
        Ka0 = 0.723848629231
       
        S0 = 3
        S1 = 1
        S2 = 1
        S3 = 1
        S4 = 1
        """
    elif modelType == 'Linear_r_i_allo':
        realModel = """
        var S1, S2, S3;
        const S0, S4;
        J0: S0 -> S1; (Kf0*S0 - Kr0*S1)/(1 + S0 + S1);
        J1: S1 -> S2; (Kf1*S1 - Kr1*S2)/(1 + S1 + S2);
        J2: S2 -> S3; (Kf2*S2 - Kr2*S3)/(1 + S2 + S3);
        J3: S3 -> S4; (Kf3*S3 - Kr3*S4)/(1 + S3 + S4 + Ki0*S1);
       
        Kf0 = 0.885822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kr0 = 0.072790656829
        Kr1 = 0.27828563882
        Kr2 = 0.166906190765
        Kr3 = 0.31057924345
        Ki0 = 0.814569278466
       
        S0 = 3
        S1 = 1
        S2 = 1
        S3 = 1
        S4 = 1
        """
    else:
        raise Exception("Requested test model not found")
        
    return realModel
