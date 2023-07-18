# -*- coding: utf-8 -*-

import tellurium as te
import random
import numpy as np

class ReactionType:
    UNIUNI = 0
    BIUNI = 1
    UNIBI = 2
    BIBI = 3

class RegulationType:
    DEFAULT = 0
    INHIBITION = 1
    ACTIVATION = 2
    INIHIBITION_ACTIVATION = 3
    
class Reversibility:
    IRREVERSIBLE = 0
    REVERSIBLE = 1

class RP22:
    UniUni = 0.5
    BiUni = 0.2
    UniBi = 0.2
    BiBI  = 0.1
    
class RP11:
    UniUni = 1
    BiUni = 0.
    UniBi = 0.
    BiBI  = 0.

class RP12:
    UniUni = 0.75
    BiUni = 0.
    UniBi = 0.25
    BiBI  = 0.
    
class RP21:
    UniUni = 0.75
    BiUni = 0.25
    UniBi = 0.
    BiBI  = 0.
    
class RLP:
    Default = 1.
    Inhib = 0#0.125
    Activ = 0#0.125
    Inhibactiv = 0#0.05
    
class REVP:
    Irreversible = 0.7
    Reversible = 0.3

#def pickRateLawType():
#    rt = np.random.random()
#    if rt < RateLawProb.default:
#        return 0
#    elif rt < RateLawProb.default + RateLawProb.inhib:
#        return 1
#    elif rt < RateLawProb.default + RateLawProb.inhib + RateLawProb.activ:
#        return 2
#    return 3
     

def pickReactionType(RP):
    rt1 = np.random.random()
    if rt1 < RP.UniUni:
        rType = ReactionType.UNIUNI
    elif (rt1 >= RP.UniUni) and (rt1 < RP.UniUni + RP.BiUni):
        rType = ReactionType.BIUNI
    elif (rt1 >= RP.UniUni + RP.BiUni) and (rt1 < RP.UniUni + RP.BiUni + RP.UniBi):
        rType = ReactionType.UNIBI
    else:
        rType = ReactionType.BIBI
    
    rt2 = np.random.random()
    if rt2 < RLP.Default:
        regType = RegulationType.DEFAULT
    elif (rt2 >= RLP.Default) and (rt2 < RLP.Default + RLP.Inhib):
        regType = RegulationType.INHIBITION
    elif (rt2 >= RLP.Default + RLP.Inhib) and (rt2 < RLP.Default + RLP.Inhib + RLP.Activ):
        regType = RegulationType.ACTIVATION
    else:
        regType = RegulationType.INIHIBITION_ACTIVATION
    
    rt3 = np.random.random()
    if rt3 < REVP.Irreversible:
        revType = Reversibility.IRREVERSIBLE
    else:
        revType = Reversibility.REVERSIBLE
        
    return rType, regType, revType


# Generates a reaction network in the form of a reaction list
# reactionList = [nSpecies, reaction, ....]
# reaction = [reactionType, [list of reactants], [list of product], rateConstant]
def generateReactionList(nsList, nrList, realFloatingIdsInd, realBoundaryIdsInd, realConcCC):

    reactionList = []
    
    # TODO: better heuristics (esp. probability)
    
    for r_idx in nrList:
        tarval = realConcCC[:,r_idx]
        ssum = np.sum(np.sign(tarval))
        if ssum > 0:
            tarsum = np.sum(tarval > 0)
            c = len(realFloatingIdsInd)-tarsum
            posRctInd = np.empty(c+len(realBoundaryIdsInd), dtype=int)
            posRctInd[:c] = realFloatingIdsInd[tarval <= 0]
            posRctInd[c:] = realBoundaryIdsInd
            posRctProb = np.empty(len(posRctInd))
            posRctProb[:c] = 1/((1+c))
            posRctProb[c:] = 1/(len(realBoundaryIdsInd)*(1+c))
            if tarsum == len(realFloatingIdsInd):
                posPrdInd = realFloatingIdsInd[tarval > 0]
                posPrdProb = tarval/np.sum(tarval)
            elif tarsum == 1 and all(tarval[tarval <= 0] == 0):
                posPrdInd = realFloatingIdsInd[tarval > 0]
                posPrdProb = [1]
            else:
                b = tarval[tarval > 0]
                a = np.sum(b)*(len(b)+1)/len(b)
                posPrdInd = np.empty(len(b)+len(realBoundaryIdsInd), dtype=int)
                posPrdInd[:len(b)] = realFloatingIdsInd[tarval > 0]
                posPrdInd[len(b):] = realBoundaryIdsInd
                posPrdProb = np.empty(len(posPrdInd))
                posPrdProb[:len(b)] = b/a
                posPrdProb[len(b):] = 1/(len(realBoundaryIdsInd)*(1+len(b)))
        elif ssum < 0:
            tarsum = np.sum(tarval < 0)
            if tarsum == len(realFloatingIdsInd):
                posRctInd = realFloatingIdsInd[tarval < 0]
                posRctProb = tarval/np.sum(tarval)
            elif tarsum == 1 and all(tarval[tarval >= 0] == 0):
                posRctInd = realFloatingIdsInd[tarval < 0]
                posRctProb = [1]
            else:
                b = tarval[tarval < 0]
                a = np.sum(b)*(len(b)+1)/len(b)
                posRctInd = np.empty(len(b)+len(realBoundaryIdsInd), dtype=int)
                posRctInd[:len(b)] = realFloatingIdsInd[tarval < 0]
                posRctInd[len(b):] = realBoundaryIdsInd
                posRctProb = np.empty(len(posRctInd))
                posRctProb[:len(b)] = b/a
                posRctProb[len(b):] = 1/(len(realBoundaryIdsInd)*(1+len(b)))
            c = len(realFloatingIdsInd)-tarsum
            posPrdInd = np.empty(c+len(realBoundaryIdsInd), dtype=int)
            posPrdInd[:c] = realFloatingIdsInd[tarval >= 0]
            posPrdInd[c:] = realBoundaryIdsInd
            posPrdProb = np.empty(len(posPrdInd))
            posPrdProb[:c] = 1/((1+c))
            posPrdProb[c:] = 1/(len(realBoundaryIdsInd)*(1+c))    
        else:
            a = realFloatingIdsInd[tarval <= 0]
            b = realFloatingIdsInd[tarval >= 0]
            posRctInd = np.empty(len(a)+len(realBoundaryIdsInd), dtype=int)
            posRctInd[:len(a)] = a
            posRctInd[len(a):] = realBoundaryIdsInd
            posPrdInd = np.empty(len(b)+len(realBoundaryIdsInd), dtype=int)
            posPrdInd[:len(b)] = b
            posPrdInd[len(b):] = realBoundaryIdsInd
            
            posRctProb = np.empty(len(posRctInd))
            posRctProb[:len(a)] = 1/((1+len(a)))
            posRctProb[len(a):] = 1/(len(realBoundaryIdsInd)*(1+len(a)))
            
            posPrdProb = np.empty(len(posPrdInd))
            posPrdProb[:len(b)] = 1/((1+len(b)))
            posPrdProb[len(b):] = 1/(len(realBoundaryIdsInd)*(1+len(b)))
        
        rct = [col[3] for col in reactionList]
        prd = [col[4] for col in reactionList]
        
        rType, regType, revType = pickReactionType(RP22)
        
        if rType == ReactionType.UNIUNI:
            # UniUni
            rct_id = np.random.choice(posRctInd, size=1, p=posRctProb).tolist()
            prd_id = np.random.choice(posPrdInd, size=1, p=posPrdProb).tolist()
            all_rct = [i for i,x in enumerate(rct) if x==rct_id]
            all_prd = [i for i,x in enumerate(prd) if x==prd_id]
            
            while (((np.any(np.isin(rct_id, realBoundaryIdsInd))) and 
                   (np.any(np.isin(prd_id, realBoundaryIdsInd)))) or 
                   (len(set(all_rct) & set(all_prd)) > 0)):
                rct_id = np.random.choice(posRctInd, size=1, p=posRctProb).tolist()
                prd_id = np.random.choice(posPrdInd, size=1, p=posPrdProb).tolist()
                # Search for potentially identical reactions
                all_rct = [i for i,x in enumerate(rct) if x==rct_id]
                all_prd = [i for i,x in enumerate(prd) if x==prd_id]
        elif rType == ReactionType.BIUNI:
            # BiUni
            rct_id = np.random.choice(posRctInd, size=2, replace=True, p=posRctProb).tolist()
            prd_id = np.random.choice(posPrdInd, size=1, p=posPrdProb).tolist()
            all_rct = [i for i,x in enumerate(rct) if set(x)==set(rct_id)]
            all_prd = [i for i,x in enumerate(prd) if x==prd_id]
            
            while (((np.any(np.isin(rct_id, realBoundaryIdsInd))) and 
                   (np.any(np.isin(prd_id, realBoundaryIdsInd)))) or 
                   (len(set(all_rct) & set(all_prd)) > 0)):
                rct_id = np.random.choice(posRctInd, size=2, replace=True, p=posRctProb).tolist()
                prd_id = np.random.choice(posPrdInd, size=1, p=posPrdProb).tolist()
                # Search for potentially identical reactions
                all_rct = [i for i,x in enumerate(rct) if set(x)==set(rct_id)]
                all_prd = [i for i,x in enumerate(prd) if x==prd_id]
        elif rType == ReactionType.UNIBI:
            # UniBi
            rct_id = np.random.choice(posRctInd, size=1, p=posRctProb).tolist()
            prd_id = np.random.choice(posPrdInd, size=2, replace=True, p=posPrdProb).tolist()
            all_rct = [i for i,x in enumerate(rct) if x==rct_id]
            all_prd = [i for i,x in enumerate(prd) if set(x)==set(prd_id)]
            
            while (((np.any(np.isin(rct_id, realBoundaryIdsInd))) and 
                   (np.any(np.isin(prd_id, realBoundaryIdsInd)))) or 
                   (len(set(all_rct) & set(all_prd)) > 0)):
                rct_id = np.random.choice(posRctInd, size=1, p=posRctProb).tolist()
                prd_id = np.random.choice(posPrdInd, size=2, replace=True, p=posPrdProb).tolist()
                # Search for potentially identical reactions
                all_rct = [i for i,x in enumerate(rct) if x==rct_id]
                all_prd = [i for i,x in enumerate(prd) if set(x)==set(prd_id)]
        else:
            # BiBi
            rct_id = np.random.choice(posRctInd, size=2, replace=True, p=posRctProb).tolist()
            prd_id = np.random.choice(posPrdInd, size=2, replace=True, p=posPrdProb).tolist()
            all_rct = [i for i,x in enumerate(rct) if set(x)==set(rct_id)]
            all_prd = [i for i,x in enumerate(prd) if set(x)==set(prd_id)]
            
            while (((np.any(np.isin(rct_id, realBoundaryIdsInd))) and 
                   (np.any(np.isin(prd_id, realBoundaryIdsInd)))) or
                   (len(set(all_rct) & set(all_prd)) > 0)):
                rct_id = np.random.choice(posRctInd, size=2, replace=True, p=posRctProb).tolist()
                prd_id = np.random.choice(posPrdInd, size=2, replace=True, p=posPrdProb).tolist()
                # Search for potentially identical reactions
                all_rct = [i for i,x in enumerate(rct) if set(x)==set(rct_id)]
                all_prd = [i for i,x in enumerate(prd) if set(x)==set(prd_id)]
                
        if regType == RegulationType.DEFAULT:
            act_id = []
            inhib_id = []
        elif regType == RegulationType.INHIBITION:
            act_id = []
            delList = np.concatenate([rct_id, prd_id])
            if len(realBoundaryIdsInd) > 0:
                delList = np.unique(np.append(delList, realBoundaryIdsInd))
            cList = np.delete(nsList, delList)
            if len(cList) == 0:
                inhib_id = []
                regType = RegulationType.DEFAULT
            else:
                inhib_id = np.random.choice(cList, size=1).tolist()
        elif regType == RegulationType.ACTIVATION:
            inhib_id = []
            delList = np.concatenate([rct_id, prd_id])
            if len(realBoundaryIdsInd) > 0:
                delList = np.unique(np.append(delList, realBoundaryIdsInd))
            cList = np.delete(nsList, delList)
            if len(cList) == 0:
                act_id = []
                regType = RegulationType.DEFAULT
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
                regType = RegulationType.DEFAULT
            else:
                reg_id = np.random.choice(cList, size=2, replace=False)
                act_id = [reg_id[0]]
                inhib_id = [reg_id[1]]
                
        reactionList.append([rType, regType, revType, rct_id, prd_id, act_id, inhib_id])
        
    return reactionList
    

# Include boundary and floating species
# Returns a list:
# [New Stoichiometry matrix, list of floatingIds, list of boundaryIds]
def getFullStoichiometryMatrix(reactionList, ns):
    
    st = np.zeros((ns, len(reactionList)), dtype=int)
    
    for index, rind in enumerate(reactionList):
        if rind[0] == ReactionType.UNIUNI:
            # UniUni
            reactant = reactionList[index][3][0]
            st[reactant, index] = st[reactant, index] - 1
            product = reactionList[index][4][0]
            st[product, index] = st[product, index] + 1
     
        elif rind[0] == ReactionType.BIUNI:
            # BiUni
            reactant1 = reactionList[index][3][0]
            st[reactant1, index] = st[reactant1, index] - 1
            reactant2 = reactionList[index][3][1]
            st[reactant2, index] = st[reactant2, index] - 1
            product = reactionList[index][4][0]
            st[product, index] = st[product, index] + 1

        elif rind[0] == ReactionType.UNIBI:
            # UniBi
            reactant1 = reactionList[index][3][0]
            st[reactant1, index] = st[reactant1, index] - 1
            product1 = reactionList[index][4][0]
            st[product1, index] = st[product1, index] + 1
            product2 = reactionList[index][4][1]
            st[product2, index] = st[product2, index] + 1

        else:
            # BiBi
            reactant1 = reactionList[index][3][0]
            st[reactant1, index] = st[reactant1, index] - 1
            reactant2 = reactionList[index][3][1]
            st[reactant2, index] = st[reactant2, index] - 1
            product1 = reactionList[index][4][0]
            st[product1, index] = st[product1, index] + 1
            product2 = reactionList[index][4][1]
            st[product2, index] = st[product2, index] + 1

    return st

def boundaryFix(stoi, realBoundaryIdsInd, nr):
    
    newboundarystoi = np.zeros((len(realBoundaryIdsInd), nr), dtype=int)
    
    vals, valr = np.nonzero(stoi[realBoundaryIdsInd])
    b = 0

    for i,j in enumerate(valr):
        if b == len(realBoundaryIdsInd):
            b = 0
        newboundarystoi[b,j] = stoi[realBoundaryIdsInd][vals[i],j]
        b += 1
    
    return newboundarystoi
    
def generateStoichiometry(signs, realFloatingIdsInd, realBoundaryIdsInd, ns, nr):
    
    stoi = np.zeros((ns, nr), dtype=int)
    rTypes = np.empty((3, nr), dtype=int)
    inhbactv = np.zeros((ns, nr), dtype=int)
    failure = np.ones(nr, dtype=int)
    
    ordering = np.argsort(np.sum(signs, axis=0))
    
    posrct = np.count_nonzero(signs<=0, axis=1)
    posprd = np.count_nonzero(signs>=0, axis=1)
    
    for r_idx in ordering:
        rcts = signs[:,r_idx] <= 0
        prds = signs[:,r_idx] >= 0
        
        rctthis = np.logical_and(posrct == 1, rcts)
        currrct = np.sum(stoi[realFloatingIdsInd] < 0, axis=1)
        rctna = np.logical_and(currrct == 0, rcts)
        prdthis = np.logical_and(posprd == 1, prds)
        currprd = np.sum(stoi[realFloatingIdsInd] > 0, axis=1)
        prdna = np.logical_and(currprd == 0, prds)
        
        stoir, rTyper, iar, f = generateSingleStoichiometry(realFloatingIdsInd, realBoundaryIdsInd, ns,
                                                           rcts, rctthis, rctna, prds, prdthis, prdna)
        stoi[:,r_idx] = stoir
        rTypes[:,r_idx] = rTyper
        inhbactv[:,r_idx] = iar
        failure[r_idx] = f
        
        posrct[rcts] -= 1
        posprd[prds] -= 1
    
    tarindx = np.count_nonzero(stoi[realBoundaryIdsInd], axis=1) == 0
    if np.any(tarindx):
        newboundarystoi = boundaryFix(stoi, realBoundaryIdsInd, nr)
        stoi[realBoundaryIdsInd] = newboundarystoi
    
    rStoi = stoi[realFloatingIdsInd]
    rStoi[rStoi > 1] = 1
    rStoi[rStoi < -1] = -1
    
    if np.sum(failure) > 0:
        return (stoi, rStoi, rTypes, inhbactv, False)
    else:
        return (stoi, rStoi, rTypes, inhbactv, True)


def generateSingleStoichiometry(realFloatingIdsInd, realBoundaryIdsInd, ns,
                                rcts, rctthis, rctna, prds, prdthis, prdna):
    
    stoir = np.zeros(ns, dtype=int)
    rTyper = np.zeros(3, dtype=int)
    inhbactv = np.zeros(ns, dtype=int)
    
    c = 0
    
    c1 = np.sum(rcts)
    c2 = np.sum(prds)
    
    if c1 == 0:
        posRctInd = realBoundaryIdsInd
        posRctProb = np.empty(len(posRctInd))
        posRctProb[:] = 1/(len(realBoundaryIdsInd))
    elif c1 == len(realFloatingIdsInd):
        posRctInd = realFloatingIdsInd
        posRctProb = np.empty(len(posRctInd))
        # a = 1/(len(realFloatingIdsInd))*np.exp(-currrct)
        posRctProb[:] = 1/(len(realFloatingIdsInd))
    else:
        posRctInd = np.empty(c1+len(realBoundaryIdsInd), dtype=int)
        posRctInd[:c1] = realFloatingIdsInd[rcts]
        posRctInd[c1:] = realBoundaryIdsInd
        posRctProb = np.empty(len(posRctInd))
        # a = 1/(1+c1)*np.exp(-currrct[realFloatingIdsInd[rcts]])
        posRctProb[:c1] = 1/(1+c1)
        posRctProb[c1:] = 1/(len(realBoundaryIdsInd)*(1+c1))
        posRctProb = posRctProb/np.sum(posRctProb)
    
    if c2 == 0:
        posPrdInd = realBoundaryIdsInd
        posPrdProb = np.empty(len(posPrdInd))
        posPrdProb[:] = 1/(len(realBoundaryIdsInd))
    elif c2 == len(realFloatingIdsInd):
        posPrdInd = realFloatingIdsInd
        posPrdProb = np.empty(len(posPrdInd))
        # a = 1/(len(realFloatingIdsInd))*np.exp(-currprd)
        posPrdProb[:] = 1/(len(realFloatingIdsInd))
    else:
        posPrdInd = np.empty(c2+len(realBoundaryIdsInd), dtype=int)
        posPrdInd[:c2] = realFloatingIdsInd[prds]
        posPrdInd[c2:] = realBoundaryIdsInd
        posPrdProb = np.empty(len(posPrdInd))
        # a = 1/(1+c2)*np.exp(-currprd[realFloatingIdsInd[prds]])
        posPrdProb[:c2] = 1/(1+c2)
        posPrdProb[c2:] = 1/(len(realBoundaryIdsInd)*(1+c2))
        posPrdProb = posPrdProb/np.sum(posPrdProb)
    
    if len(posRctInd) == len(posPrdInd) == 1:
        rType, regType, revType = pickReactionType(RP11)
    elif len(posRctInd) == 1 and len(posPrdInd) == 2:
        rType, regType, revType = pickReactionType(RP12)
    elif len(posRctInd) == 2 and len(posPrdInd) == 1:
        rType, regType, revType = pickReactionType(RP21)
    else:
        rType, regType, revType = pickReactionType(RP22)
    
    if rType == ReactionType.UNIUNI:
        # UniUni
        if np.logical_and(rctthis, rctna).any():
            rct_id = realFloatingIdsInd[np.logical_and(rcts, np.logical_and(rctthis, rctna))]
            rct_id = np.random.choice(rct_id, size=1)
        else:
            rct_id = np.random.choice(posRctInd, size=1, p=posRctProb)
        if np.logical_and(prdthis, prdna).any():
            prd_id = realFloatingIdsInd[np.logical_and(prds, np.logical_and(prdthis, prdna))]
            prd_id = np.random.choice(prd_id, size=1)
        else:
            prd_id = np.random.choice(posPrdInd, size=1, p=posPrdProb)
        
        while (c<100 and (((np.any(np.isin(rct_id, realBoundaryIdsInd))) and 
               (np.any(np.isin(prd_id, realBoundaryIdsInd)))) or
               (rct_id.tolist() == prd_id.tolist()))):
            if np.logical_and(rctthis, rctna).any():
                rct_id = realFloatingIdsInd[np.logical_and(rcts, np.logical_and(rctthis, rctna))]
                rct_id = np.random.choice(rct_id, size=1)
            else:
                rct_id = np.random.choice(posRctInd, size=1, p=posRctProb)
            if np.logical_and(prdthis, prdna).any():
                prd_id = realFloatingIdsInd[np.logical_and(prds, np.logical_and(prdthis, prdna))]
                prd_id = np.random.choice(prd_id, size=1)
            else:
                prd_id = np.random.choice(posPrdInd, size=1, p=posPrdProb)
            c += 1
    
    if rType == ReactionType.BIUNI:
        # BiUni
        if np.logical_and(rctthis, rctna).any():
            rct_id = realFloatingIdsInd[np.logical_and(rcts, np.logical_and(rctthis, rctna))]
            rct_id = np.random.choice(rct_id, size=1)
            posRctIndr = np.delete(posRctInd, rct_id)
            posRctProbr = np.delete(posRctProb, rct_id)
            posRctProbr = posRctProbr/np.sum(posRctProbr)
            rct_id = np.append(rct_id, np.random.choice(posRctIndr, size=1, p=posRctProbr))
        else:
            rct_id = np.random.choice(posRctInd, size=2, p=posRctProb)
        if np.logical_and(prdthis, prdna).any():
            prd_id = realFloatingIdsInd[np.logical_and(prds, np.logical_and(prdthis, prdna))]
            prd_id = np.random.choice(prd_id, size=1)
        else:
            prd_id = np.random.choice(posPrdInd, size=1, p=posPrdProb)
        
        while (c<100 and (((np.any(np.isin(rct_id, realBoundaryIdsInd))) and 
               (np.any(np.isin(prd_id, realBoundaryIdsInd)))) or
               (len(set(rct_id) & set(prd_id))>0))):
            if np.logical_and(rctthis, rctna).any():
                rct_id = realFloatingIdsInd[np.logical_and(rcts, np.logical_and(rctthis, rctna))]
                rct_id = np.random.choice(rct_id, size=1)
                posRctIndr = np.delete(posRctInd, rct_id)
                posRctProbr = np.delete(posRctProb, rct_id)
                posRctProbr = posRctProbr/np.sum(posRctProbr)
                rct_id = np.append(rct_id, np.random.choice(posRctIndr, size=1, p=posRctProbr))
            else:
                rct_id = np.random.choice(posRctInd, size=2, p=posRctProb)
            if np.logical_and(prdthis, prdna).any():
                prd_id = realFloatingIdsInd[np.logical_and(prds, np.logical_and(prdthis, prdna))]
                prd_id = np.random.choice(prd_id, size=1)
            else:
                prd_id = np.random.choice(posPrdInd, size=1, p=posPrdProb)
            c += 1
    
    if rType == ReactionType.UNIBI:
        # UniBi
        if np.logical_and(rctthis, rctna).any():
            rct_id = realFloatingIdsInd[np.logical_and(rcts, np.logical_and(rctthis, rctna))]
            rct_id = np.random.choice(rct_id, size=1)
        else:
            rct_id = np.random.choice(posRctInd, size=1, p=posRctProb)
        if np.logical_and(prdthis, prdna).any():
            prd_id = realFloatingIdsInd[np.logical_and(prds, np.logical_and(prdthis, prdna))]
            prd_id = np.random.choice(prd_id, size=1)
            posPrdIndr = np.delete(posPrdInd, prd_id)
            posPrdProbr = np.delete(posPrdProb, prd_id)
            posPrdProbr = posPrdProbr/np.sum(posPrdProbr)
            prd_id = np.append(prd_id, np.random.choice(posPrdIndr, size=1, p=posPrdProbr))
        else:
            prd_id = np.random.choice(posPrdInd, size=2, p=posPrdProb)
        
        while (c<100 and (((np.any(np.isin(rct_id, realBoundaryIdsInd))) and 
               (np.any(np.isin(prd_id, realBoundaryIdsInd)))) or
               (len(set(rct_id) & set(prd_id))>0))):
            if np.logical_and(rctthis, rctna).any():
                rct_id = realFloatingIdsInd[np.logical_and(rcts, np.logical_and(rctthis, rctna))]
                rct_id = np.random.choice(rct_id, size=1)
            else:
                rct_id = np.random.choice(posRctInd, size=1, p=posRctProb)
            if np.logical_and(prdthis, prdna).any():
                prd_id = realFloatingIdsInd[np.logical_and(prds, np.logical_and(prdthis, prdna))]
                prd_id = np.random.choice(prd_id, size=1)
                posPrdIndr = np.delete(posPrdInd, prd_id)
                posPrdProbr = np.delete(posPrdProb, prd_id)
                posPrdProbr = posPrdProbr/np.sum(posPrdProbr)
                prd_id = np.append(prd_id, np.random.choice(posPrdIndr, size=1, p=posPrdProbr))
            else:
                prd_id = np.random.choice(posPrdInd, size=2, p=posPrdProb, replace=True)
            c += 1
            
    if rType == ReactionType.BIBI:
        # BiBi
        if np.logical_and(rctthis, rctna).any():
            rct_id = realFloatingIdsInd[np.logical_and(rcts, np.logical_and(rctthis, rctna))]
            rct_id = np.random.choice(rct_id, size=1)
            posRctIndr = np.delete(posRctInd, rct_id)
            posRctProbr = np.delete(posRctProb, rct_id)
            posRctProbr = posRctProbr/np.sum(posRctProbr)
            rct_id = np.append(rct_id, np.random.choice(posRctIndr, size=1, p=posRctProbr))
        else:
            rct_id = np.random.choice(posRctInd, size=2, p=posRctProb, replace=True)
        if np.logical_and(prdthis, prdna).any():
            prd_id = realFloatingIdsInd[np.logical_and(prds, np.logical_and(prdthis, prdna))]
            prd_id = np.random.choice(prd_id, size=1)
            posPrdIndr = np.delete(posPrdInd, prd_id)
            posPrdProbr = np.delete(posPrdProb, prd_id)
            posPrdProbr = posPrdProbr/np.sum(posPrdProbr)
            prd_id = np.append(prd_id, np.random.choice(posPrdIndr, size=1, p=posPrdProbr))
        else:
            prd_id = np.random.choice(posPrdInd, size=2, p=posPrdProb)
        
        while (c<100 and (((np.any(np.isin(rct_id, realBoundaryIdsInd))) and 
               (np.any(np.isin(prd_id, realBoundaryIdsInd)))) or
               (len(set(rct_id) & set(prd_id))>1)) or (rct_id.tolist() == prd_id.tolist())):
            if np.logical_and(rctthis, rctna).any():
                rct_id = realFloatingIdsInd[np.logical_and(rcts, np.logical_and(rctthis, rctna))]
                rct_id = np.random.choice(rct_id, size=1)
                posRctIndr = np.delete(posRctInd, rct_id)
                posRctProbr = np.delete(posRctProb, rct_id)
                posRctProbr = posRctProbr/np.sum(posRctProbr)
                rct_id = np.append(rct_id, np.random.choice(posRctIndr, size=1, p=posRctProbr))
            else:
                rct_id = np.random.choice(posRctInd, size=2, p=posRctProb, replace=True)
            if np.logical_and(prdthis, prdna).any():
                prd_id = realFloatingIdsInd[np.logical_and(prds, np.logical_and(prdthis, prdna))]
                prd_id = np.random.choice(prd_id, size=1)
                posPrdIndr = np.delete(posPrdInd, prd_id)
                posPrdProbr = np.delete(posPrdProb, prd_id)
                posPrdProbr = posPrdProbr/np.sum(posPrdProbr)
                prd_id = np.append(prd_id, np.random.choice(posPrdIndr, size=1, p=posPrdProbr))
            else:
                prd_id = np.random.choice(posPrdInd, size=2, p=posPrdProb, replace=True)
            c += 1
    
    for s in rct_id:
        stoir[s] -= 1
    for s in prd_id:
        stoir[s] += 1
        
    if regType == RegulationType.DEFAULT:
        pass
    elif regType == RegulationType.INHIBITION:
        delList = rct_id.tolist() + prd_id.tolist() + realBoundaryIdsInd.tolist()
        delList = np.unique(delList)
        cList = np.delete(np.arange(ns), delList)
        if len(cList) == 0:
            regType = RegulationType.DEFAULT
        else:
            inhib_id = np.random.choice(cList, size=1).tolist()
            inhbactv[inhib_id] = -1
    elif regType == RegulationType.ACTIVATION:
        delList = rct_id.tolist() + prd_id.tolist() + realBoundaryIdsInd.tolist()
        delList = np.unique(delList)
        cList = np.delete(np.arange(ns), delList)
        if len(cList) == 0:
            regType = RegulationType.DEFAULT
        else:
            act_id = np.random.choice(cList, size=1).tolist()
            inhbactv[act_id] = 1
    else:
        delList = rct_id.tolist() + prd_id.tolist() + realBoundaryIdsInd.tolist()
        delList = np.unique(delList)
        cList = np.delete(np.arange(ns), delList)
        if len(cList) < 2:
            regType = RegulationType.DEFAULT
        else:
            reg_id = np.random.choice(cList, size=2, replace=False)
            inhbactv[reg_id[0]] = 1
            inhbactv[reg_id[1]] = -1
    
    rTyper[0] = rType
    rTyper[1] = regType
    rTyper[2] = revType
    
    f  = 0
    if c == 100:
        f = 1

    return stoir, rTyper, inhbactv, f


# Removes boundary or orphan species from stoichiometry matrix
def removeBoundaryNodes(st, nsList, nrList):
    
    indexes = []
    orphanSpecies = []
    countBoundarySpecies = 0
    for r in nsList: 
        # Scan across the columns, count + and - coefficients
        plusCoeff = 0; minusCoeff = 0
        for c in nrList:
            if st[r,c] < 0:
                minusCoeff = minusCoeff + 1
            elif st[r,c] > 0:
                plusCoeff = plusCoeff + 1
        if plusCoeff == 0 and minusCoeff == 0:
            # No reaction attached to this species
            orphanSpecies.append(r)
        elif plusCoeff == 0 and minusCoeff != 0:
            # Species is a source
            indexes.append(r)
            countBoundarySpecies = countBoundarySpecies + 1
        elif minusCoeff == 0 and plusCoeff != 0:
            # Species is a sink
            indexes.append(r)
            countBoundarySpecies = countBoundarySpecies + 1

    floatingIds = np.delete(nsList, indexes+orphanSpecies, axis=0).astype(int)
    boundaryIds = indexes
    
    rsm = st[floatingIds]
    rsm[rsm>1] = 1
    rsm[rsm<-1] = -1
    
    floatingIds = floatingIds.tolist()
    
    return rsm, floatingIds, boundaryIds


def generateRateLaw(rl, floatingIds, boundaryIds, rlt, Jind):
    
    Klist = []
    
    T = ''
    D = ''
    Rreg = ''
    Dreg = ''
    
    # T
    T = T + '(Kf_' + str(Jind) + '*'
    Klist.append('Kf_' + str(Jind))
    Klist.append('h_' + str(Jind))
    
    T = T + '('
    for i in range(len(rl[Jind][1])):
        T = T + '(S' + str(rl[Jind][1][i]) + '/Km_' + str(rl[Jind][1][i]) + ')^h_' + str(Jind)
        Klist.append('Km_' + str(rl[Jind][1][i]))
        if i < len(rl[Jind][1]) - 1:
            T = T + '*'
    T = T + ')'
    
    T = T + '-Kr_' + str(Jind) + '*'
    Klist.append('Kr_' + str(Jind))
    
    T = T + '('
    for i in range(len(rl[Jind][2])):
        T = T + '(S' + str(rl[Jind][2][i]) + '/Km_' + str(rl[Jind][2][i]) +')^h_' + str(Jind)
        Klist.append('Km_' + str(rl[Jind][2][i]))
        if i < len(rl[Jind][2]) - 1:
            T = T + '*'
            
    T = T + '))'
        
    # D
    D = D + '('
    
    for i in range(len(rl[Jind][1])):
        D = D + '((1 + (S' + str(rl[Jind][1][i]) + '/Km_' + str(rl[Jind][1][i]) + '))^h_' + str(Jind) + ')'
        Klist.append('Km_' + str(rl[Jind][1][i]))
        if i < len(rl[Jind][1]) - 1:
            D = D + '*'
    
    D = D + '+'
    
    for i in range(len(rl[Jind][2])):
        D = D + '((1 + (S' + str(rl[Jind][2][i]) + '/Km_' + str(rl[Jind][2][i]) + '))^h_' + str(Jind) + ')'
        Klist.append('Km_' + str(rl[Jind][2][i]))
        if i < len(rl[Jind][2]) - 1:
            D = D + '*'
    
    D = D + '-1)'
        
    #Rreg
    if (rlt == 1) or (rlt == 3):
        pass
    
    #Dreg
    if (rlt == 2) or (rlt == 3):
        pass
    
    
    rateLaw = Rreg + T + '/(' + D +  Dreg + ')'
        
    return rateLaw, Klist


def generateSimpleRateLaw(rind, Jind, real, tar):
    
    Klist = []
    
    T = ''
    D = ''
    ACT = ''
    INH = ''
    
    # T
    T = T + '(Kf' + str(Jind) + '*'
    Klist.append('Kf' + str(Jind))
    
    for i in range(len(rind[3])):
        T = T + real[tar.index(rind[3][i])]
        if i < len(rind[3]) - 1:
            T = T + '*'
    
    if rind[2] == Reversibility.REVERSIBLE:
        T = T + ' - Kr' + str(Jind) + '*'
        Klist.append('Kr' + str(Jind))
        
        for i in range(len(rind[4])):
            T = T + real[tar.index(rind[4][i])]
            if i < len(rind[4]) - 1:
                T = T + '*'
            
    T = T + ')'
        
    # D
    D = D + '1 + '
    
    for i in range(len(rind[3])):
        D = D + real[tar.index(rind[3][i])]
        if i < len(rind[3]) - 1:
            D = D + '*'
    
    if rind[2] == Reversibility.REVERSIBLE:
        D = D + ' + '
        for i in range(len(rind[4])):
            D = D + real[tar.index(rind[4][i])]
            if i < len(rind[4]) - 1:
                D = D + '*'
    
    # Activation
    if (len(rind[5]) > 0):
        for i in range(len(rind[5])):
            ACT = ACT + '(1 + Ka' + str(Jind) + str(i) + '*'
            Klist.append('Ka' + str(Jind) + str(i))
            ACT = ACT + real[tar.index(rind[5][i])] + ')*'
            
    # Inhibition
    if (len(rind[6]) > 0):
        for i in range(len(rind[6])):
            INH = INH + '(1/(1 + Ki' + str(Jind) + str(i) + '*'
            Klist.append('Ki' + str(Jind) + str(i))
            INH = INH + real[tar.index(rind[6][i])] + '))*'
    
    rateLaw = ACT + INH + T + '/(' + D + ')'
        
    return rateLaw, Klist


def generateAntimony(floatingIds, boundaryIds, fid, bid, reactionList, boundary_init=None):
    Klist = []
    
    real = floatingIds + boundaryIds
    tar = fid + bid
    
    # List species
    antStr = ''
    if len(floatingIds) > 0:
        antStr = antStr + 'var ' + str(floatingIds[0])
        for index in floatingIds[1:]:
            antStr = antStr + ', ' + str(index)
        antStr = antStr + ';\n'
    
    if len(boundaryIds) > 0:
        antStr = antStr + 'const ' + str(boundaryIds[0])
        for index in boundaryIds[1:]:
            antStr = antStr + ', ' + str(index)
        antStr = antStr + ';\n'

    # List reactions
    for index, rind in enumerate(reactionList):
        if rind[0] == ReactionType.UNIUNI:
            # UniUni
            antStr = antStr + 'J' + str(index) + ': ' + real[tar.index(rind[3][0])]
            antStr = antStr + ' -> '
            antStr = antStr + real[tar.index(rind[4][0])]
            antStr = antStr + '; '
            RateLaw, klist_i = generateSimpleRateLaw(rind, index, real, tar)
            antStr = antStr + RateLaw
            Klist.append(klist_i)
        elif rind[0] == ReactionType.BIUNI:
            # BiUni
            antStr = antStr + 'J' + str(index) + ': ' + real[tar.index(rind[3][0])]
            antStr = antStr + ' + '
            antStr = antStr + real[tar.index(rind[3][1])]
            antStr = antStr + ' -> '
            antStr = antStr + real[tar.index(rind[4][0])]
            antStr = antStr + '; '
            RateLaw, klist_i = generateSimpleRateLaw(rind, index, real, tar)
            antStr = antStr + RateLaw
            Klist.append(klist_i)
        elif rind[0] == ReactionType.UNIBI:
            # UniBi
            antStr = antStr + 'J' + str(index) + ': ' + real[tar.index(rind[3][0])]
            antStr = antStr + ' -> '
            antStr = antStr + real[tar.index(rind[4][0])]
            antStr = antStr + ' + '
            antStr = antStr + real[tar.index(rind[4][1])]
            antStr = antStr + '; '
            RateLaw, klist_i = generateSimpleRateLaw(rind, index, real, tar)
            antStr = antStr + RateLaw
            Klist.append(klist_i)
        else:
            # BiBi
            antStr = antStr + 'J' + str(index) + ': ' + real[tar.index(rind[3][0])]
            antStr = antStr + ' + '
            antStr = antStr + real[tar.index(rind[3][1])]
            antStr = antStr + ' -> '
            antStr = antStr + real[tar.index(rind[4][0])]
            antStr = antStr + ' + '
            antStr = antStr + real[tar.index(rind[4][1])]
            antStr = antStr + '; '
            RateLaw, klist_i = generateSimpleRateLaw(rind, index, real, tar)
            antStr = antStr + RateLaw
            Klist.append(klist_i)
        antStr = antStr + ';\n'

    # List rate constants
    antStr = antStr + '\n'
    Klist_f = [item for sublist in Klist for item in sublist]
    
    for i in range(len(Klist_f)):
        if Klist_f[i].startswith('Kf'):
            antStr = antStr + Klist_f[i] + ' = 0.5\n'
        elif Klist_f[i].startswith('Kr'):
            antStr = antStr + Klist_f[i] + ' = 0.25\n'
        elif Klist_f[i].startswith('Ka'):
            antStr = antStr + Klist_f[i] + ' = 0.5\n'
        elif Klist_f[i].startswith('Ki'):
            antStr = antStr + Klist_f[i] + ' = 0.5\n'
        
    # Initialize boundary species
    antStr = antStr + '\n'
    if type(boundary_init) == type(None):
        for index, bind in enumerate(boundaryIds):
            antStr = antStr + str(bind) + ' = ' + str(np.random.randint(1,6)) + '\n'
    else:
        for index, bind in enumerate(boundaryIds):
            antStr = antStr + str(bind) + ' = ' + str(boundary_init[index]) + '\n'
    
    # Initialize floating species
    for index, find in enumerate(floatingIds):
        antStr = antStr + str(find) + ' = ' + '1\n'
        
    return antStr
     

def generateSimpleRateLawStoich(regTypes, revTypes, Jind, rct, prd, inhibactiv, real):
    
    Klist = []
    
    T = ''
    D = ''
    ACT = ''
    INH = ''
    
    # T
    T = T + '(Kf{}*'.format(Jind)
    Klist.append('Kf{}'.format(Jind))
    
    for i,j in enumerate(rct):
        T = T + '{}'.format(j)
        if i < len(rct) - 1:
            T = T + '*'
    
    if revTypes == Reversibility.REVERSIBLE:
        T = T + ' - Kr{}*'.format(Jind)
        Klist.append('Kr{}'.format(Jind))
        
        for i,j in enumerate(prd):
            T = T + '{}'.format(j)
            if i < len(prd) - 1:
                T = T + '*'
            
    T = T + ')'
        
    # D
    D = D + '1 + '
    
    for i,j in enumerate(rct):
        D = D + '{}'.format(j)
        if i < len(rct) - 1:
            D = D + '*'
    
    if revTypes == Reversibility.REVERSIBLE:
        D = D + ' + '
        for i,j in enumerate(prd):
            D = D + '{}'.format(j)
            if i < len(prd) - 1:
                D = D + '*'
    
    # Activation
    if regTypes == RegulationType.ACTIVATION:
        inhibactiv.T[Jind]
        act = real[inhibactiv<0]
        for i,j in enumerate(act):
            ACT = ACT + '(1 + Ka{}{}*'.format(Jind, i)
            Klist.append('Ka{}{}*'.format(Jind, i))
            ACT = ACT + '{})*'.format(j)
            
    # Inhibition
    if regTypes == RegulationType.INHIBITION:
        inh = real[inhibactiv>0]
        for i,j in enumerate(inh):
            INH = INH + '(1/(1 + Ki{}{}*'.format(Jind, i)
            Klist.append('Ki{}{}*'.format(Jind, i))
            INH = INH + '{}))*'.format(j)
    
    rateLaw = '{}{}{}/({})'.format(ACT, INH, T, D)
        
    return rateLaw, Klist

def generateAntfromStoich(realFloatingIds, realBoundaryIds, st, 
                          rType, inhibactiv, boundary_init=None):
    Klist = []
    
    real = np.array(realFloatingIds + realBoundaryIds)
    
    # List species
    antStr = ''
    antStr = antStr + 'var {}'.format(realFloatingIds[0])
    for index in realFloatingIds[1:]:
        antStr = antStr + ', {}'.format(index)
    antStr = antStr + ';\n'
    
    antStr = antStr + 'const {}'.format(realBoundaryIds[0])
    for index in realBoundaryIds[1:]:
        antStr = antStr + ', {}'.format(index)
    antStr = antStr + ';\n'

    # List reactions
    for index, rind in enumerate(st.T):
        rct = real[rind<0]
        prd = real[rind>0]
        if rType[0][index] == ReactionType.UNIUNI:
            # UniUni
            antStr = antStr + 'J{}: {} -> {}; '.format(index, rct[0], prd[0])
            RateLaw, klist_i = generateSimpleRateLawStoich(rType[1][index], rType[2][index], 
                                                           index, rct, prd, inhibactiv, real)
            antStr = antStr + RateLaw
            Klist.append(klist_i)
        elif rType[0][index] == ReactionType.BIUNI:
            # BiUni
            if len(rct) == 1:
                rct = np.repeat(rct, 2)
            antStr = antStr + 'J{}: {} + {} -> {}; '.format(index, rct[0], rct[1], prd[0])
            RateLaw, klist_i = generateSimpleRateLawStoich(rType[1][index], rType[2][index], 
                                                           index, rct, prd, inhibactiv, real)
            antStr = antStr + RateLaw
            Klist.append(klist_i)
        elif rType[0][index] == ReactionType.UNIBI:
            # UniBi
            if len(prd) == 1:
                prd = np.repeat(prd, 2)
            antStr = antStr + 'J{}: {} -> {} + {}; '.format(index, rct[0], prd[0], prd[1])
            RateLaw, klist_i = generateSimpleRateLawStoich(rType[1][index], rType[2][index], 
                                                           index, rct, prd, inhibactiv, real)
            antStr = antStr + RateLaw
            Klist.append(klist_i)
        else:
            # BiBi
            if len(rct) == 1:
                rct = np.repeat(rct, 2)
            if len(prd) == 1:
                prd = np.repeat(prd, 2)
            antStr = antStr + 'J{}: {} + {} -> {} + {}; '.format(index, rct[0], 
                                                                    rct[1], prd[0], prd[1])
            RateLaw, klist_i = generateSimpleRateLawStoich(rType[1][index], rType[0][index], 
                                                           index, rct, prd, inhibactiv, real)
            antStr = antStr + RateLaw
            Klist.append(klist_i)
        antStr = antStr + ';\n'

    # List rate constants
    antStr = antStr + '\n'
    Klist_f = [item for sublist in Klist for item in sublist]
    
    for i in range(len(Klist_f)):
        if Klist_f[i].startswith('Kf'):
            antStr = antStr + Klist_f[i] + ' = 0.5\n'
        elif Klist_f[i].startswith('Kr'):
            antStr = antStr + Klist_f[i] + ' = 0.25\n'
        elif Klist_f[i].startswith('Ka'):
            antStr = antStr + Klist_f[i] + ' = 0.5\n'
        elif Klist_f[i].startswith('Ki'):
            antStr = antStr + Klist_f[i] + ' = 0.5\n'
        
    # Initialize boundary species
    antStr = antStr + '\n'
    if type(boundary_init) == type(None):
        for index, bind in enumerate(realBoundaryIds):
            antStr = antStr + '{} = {}\n'.format(bind, np.random.randint(1,6))
    else:
        for index, bind in enumerate(realBoundaryIds):
            antStr = antStr + '{} = {}\n'.format(bind, boundary_init[index])
    
    # Initialize floating species
    for index, find in enumerate(realFloatingIds):
        antStr = antStr + '{} = 1\n'.format(find)
        
    return antStr


def generateParameterBoundary(glgp):
    
    pBound = []
    
    for i in range(len(glgp)):
        if glgp[i].startswith('Kf'):
            pBound.append((1e-2, 1.))
        elif glgp[i].startswith('Kr'):
            pBound.append((1e-2, 1.))
        elif glgp[i].startswith('Ka'):
            pBound.append((1e-2, 1.))
        elif glgp[i].startswith('Ki'):
            pBound.append((1e-2, 1.))

    return pBound
    

def generateLinearChainAnt(ns):
    order = np.random.sample(range(ns), ns)
    
    antStr = ''
    antStr = antStr + 'var S' + str(order[1])
    
    for i in range(ns - 3):
        antStr = antStr + ', S' + str(order[i + 2])
    
    antStr = antStr + '\n'
    antStr = antStr + 'const S' + str(order[0]) + ', S' + str(order[-1])
    antStr = antStr + ';\n'
    
    for i in range(ns - 1):
        antStr = antStr + 'S' + str(order[i]) + ' -> S' + str(order[i + 1]) + '; k' + str(order[i]) + '*S' + str(order[i]) + '\n'
    
    antStr = antStr + '\n'
    antStr = antStr + 'S' + str(order[0]) + ' = ' + str(random.randint (1,6)) + ';\n'
    
    for i in range(ns - 1):
        antStr = antStr + 'k' + str(order[i]) + ' = ' + str(random.random()) + ';\n'
        
    return antStr


# TODO: generate reaction list from aliases
def generateReactionListFromAntimony(antStr):
    """
    Generate reaction list from a model encoded in Antimony
    
    :param antStr: model encoded in Antimony
    """
    import libsbml
    import sympy
    
    r = te.loada(antStr)
    
    numBnd = r.getNumBoundarySpecies()
    numFlt = r.getNumFloatingSpecies()
    boundaryId = r.getBoundarySpeciesIds()
    floatingId = r.getFloatingSpeciesIds()
    nr = r.getNumReactions()
    
    # prepare symbols for sympy
    boundaryId_sympy = [] 
    floatingId_sympy = []
    
    # Fix issues with reserved characters
    for i in range(numBnd):
        if boundaryId[i] == 'S':
            boundaryId_sympy.append('_S')
        else:
            boundaryId_sympy.append(boundaryId[i])
    
    for i in range(numFlt):
        if floatingId[i] == 'S':
            floatingId_sympy.append('_S')
        else:
            floatingId_sympy.append(floatingId[i])
    
    paramIdsStr = ' '.join(r.getGlobalParameterIds())
    floatingIdsStr = ' '.join(floatingId_sympy)
    boundaryIdsStr = ' '.join(boundaryId_sympy)
    comparmentIdsStr = ' '.join(r.getCompartmentIds())
    
    allIds = paramIdsStr + ' ' + floatingIdsStr + ' ' + boundaryIdsStr + ' ' + comparmentIdsStr
    
    avsym = sympy.symbols(allIds)
    
    # extract reactant, product, modifiers, and kinetic laws
    rct = []
    prd = []
    mod = []
    r_type = []
    kineticLaw = []
    mod_type = []
    
    doc = libsbml.readSBMLFromString(r.getSBML())
    sbmlmodel = doc.getModel()

    for slr in sbmlmodel.getListOfReactions():
        temprct = []
        tempprd = []
        tempmod = []
        
        sbmlreaction = sbmlmodel.getReaction(slr.getId())
        for sr in range(sbmlreaction.getNumReactants()):
            sbmlrct = sbmlreaction.getReactant(sr)
            temprct.append(sbmlrct.getSpecies())
        for sp in range(sbmlreaction.getNumProducts()):
            sbmlprd = sbmlreaction.getProduct(sp)
            tempprd.append(sbmlprd.getSpecies())
        for sm in range(sbmlreaction.getNumModifiers()):
            sbmlmod = sbmlreaction.getModifier(sm)
            tempmod.append(sbmlmod.getSpecies())
        kl = sbmlreaction.getKineticLaw()
        
        rct.append(sorted(temprct, key=lambda v: (v.upper(), v[0].islower())))
        prd.append(sorted(tempprd, key=lambda v: (v.upper(), v[0].islower())))
        mod.append(sorted(tempmod, key=lambda v: (v.upper(), v[0].islower())))
        
        # Update kinetic law according to change in species name
        kl_split = kl.getFormula().split(' ')
        for i in range(len(kl_split)):
            if kl_split[i] == 'S':
                kl_split[i] = '_S'
        
        kineticLaw.append(' '.join(kl_split))
    
    # use sympy for analyzing modifiers weSmart
    for ml in range(len(mod)):
        mod_type_temp = []
        expression = kineticLaw[ml]
        n,d = sympy.fraction(expression)
        for ml_i in range(len(mod[ml])):
            if n.has(mod[ml][ml_i]) and not d.has(mod[ml][ml_i]):
                mod_type_temp.append('activator')
            elif d.has(mod[ml][ml_i]) and not n.has(mod[ml][ml_i]):
                mod_type_temp.append('inhibitor')
            elif n.has(mod[ml][ml_i]) and d.has(mod[ml][ml_i]):
                mod_type_temp.append('inhibitor_activator')
            else:
                mod_type_temp.append('modifier')
        mod_type.append(mod_type_temp)
        
        # In case all products are in rate law, assume it is a reversible reaction
        if all(ext in str(n) for ext in prd[ml]):
            r_type.append('reversible')
        else:
            r_type.append('irreversible')
        
    reactionList = []
    
    for i in range(nr):
        inhib = []
        activ = []
        rct_temp = []
        prd_temp = []
        
        if len(rct[i]) == 1:
            if len(prd[i]) == 1:
                rType = 0
            elif len(prd[i]) == 2:
                rType = 2
        elif len(rct[i]) == 2:
            if len(prd[i]) == 1:
                rType = 1
            elif len(prd[i]) == 2:
                rType = 3
        
        for j in range(len(rct[i])):
            rct_temp.append(int(rct[i][j][1:]))
            
        for j in range(len(prd[i])):
            prd_temp.append(int(prd[i][j][1:]))
        
        if len(mod_type[i]) == 0:
            regType = 0
        else:
            for k in range(len(mod_type[i])):
                if mod_type[i][k] == 'inhibitor':
                    inhib.append(int(mod[i][k][1:]))
                elif mod_type[i][k] == 'activator':
                    activ.append(int(mod[i][k][1:]))
                
                if len(inhib) > 0:
                    if len(activ) == 0:
                        regType = 1
                    else:
                        regType = 3
                else:
                    regType = 2
                
        if r_type[i] == 'reversible':
            revType = 1
        else:
            revType = 0

        reactionList.append([rType, regType, revType, rct_temp, prd_temp, activ,
                             inhib])
    
    return reactionList

