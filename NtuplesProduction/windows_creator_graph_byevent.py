from __future__ import print_function
from math import pi, sqrt, cosh, log
import random
import string
from collections import OrderedDict, defaultdict
from operator import itemgetter, attrgetter
import calo_association
import random
from pprint import pprint
import json
import numpy as np
import ROOT as R
R.gROOT.ProcessLine(".L Mustache.C+")
import correctionlib

'''
This script extracts the windows and associated clusters from events
coming from RecoSimDumper. 

All windows are created:  seeds inside other windows creates their window
'''


def DeltaR(phi1, eta1, phi2, eta2):
    dphi = phi1 - phi2
    if dphi > pi: dphi -= 2*pi
    if dphi < -pi: dphi += 2*pi
    deta = eta1 - eta2
    deltaR = (deta*deta) + (dphi*dphi)
    return sqrt(deltaR)

def DeltaPhi(phi1, phi2):
    dphi = phi1 - phi2
    if dphi > pi: dphi -= 2*pi
    if dphi < -pi: dphi += 2*pi
    return dphi

def is_in_geom_mustache(seed_eta, seed_phi, cl_eta, cl_phi, cl_en ):
    '''
    This functions associates a cluster as true matched only if it is in the mustache
    and if it passes a threshold in simfraction
    '''
    is_in_mustache = False 
    if R.inMustache(seed_eta, seed_phi, cl_en,cl_eta,cl_phi):
        if R.inDynamicDPhiWindow(seed_eta, seed_phi, cl_en, cl_eta, cl_phi):
            is_in_mustache = True
    return is_in_mustache

# Check if a xtal is in the window
def in_window(seed_eta, seed_phi, seed_iz, eta, phi, iz, window_deta_up, windows_deta_down, window_dphi):
    if seed_iz != iz: return False, (-1,-1)
    # Delta Eta ordering
    etaw = eta - seed_eta
    if seed_eta < 0:
        etaw = -etaw
    phiw = DeltaPhi(seed_phi, phi)
    if etaw >= windows_deta_down and etaw <= window_deta_up  and abs(phiw) <= window_dphi: 
        return True, (etaw, phiw)
    else:
        return False,(-1,-1)

def get_cluster_hits(pfclhit_ieta,pfclhit_iphi,pfclhit_iz,pfclhit_energy, pfclhit_fraction):
    #ieta,iphi,iz,rechit,fraction
    data = []
    for i in range(len(pfclhit_ieta)):
        el = [] 
        el.append(pfclhit_ieta[i])
        el.append(pfclhit_iphi[i])
        el.append(pfclhit_iz[i])
        el.append(pfclhit_energy[i])  # total rechit
        el.append(pfclhit_energy[i]*pfclhit_fraction[i]) #fraction of rechit energy associated to this pfcluster
        el.append(pfclhit_fraction[i]) # fraction
        data.append(el)
    return data


class WindowCreator():

    def __init__(self, simfraction_thresholds,  seed_min_fraction=1e-2, cl_min_fraction=1e-4, simenergy_pu_limit = 1.5,
                 min_et_seed=1.,  max_et_seed=1e10, max_et_isolated_cl=1e10, assoc_strategy="sim_fraction",  nocalomatchedNmax=0,  do_pu_sim=False):
        self.seed_min_fraction = seed_min_fraction
        self.cluster_min_fraction = cl_min_fraction
        self.simfraction_thresholds = correctionlib.CorrectionSet.from_file(simfraction_thresholds)["simfraction_thres"]
        self.simenergy_pu_limit = simenergy_pu_limit
        self.min_et_seed=min_et_seed
        self.max_et_seed=max_et_seed
        self.max_et_isolated_cl = max_et_isolated_cl
        self.assoc_strategy = assoc_strategy
        self.do_pu_sim = do_pu_sim
        self.nocalomatchedNmax = nocalomatchedNmax


        
    def remove_clusters(self, idxs_to_remove, nodes_features,
                            nodes_sim_features, edges_idx, edges_labels):
            # Remove nodes without connections
            for icl in idxs_to_remove:
                for key in nodes_features.keys():
                    nodes_features[key][icl] = None
                    for key in nodes_sim_features.keys():
                        nodes_sim_features[key][icl] = None

            # Create a mapping from old indices to new indices
            old_to_new_index = {}
            new_index = 0
            for old_index in range(len(nodes_features["cl_en"])):
                if nodes_features["cl_en"][old_index] is not None:
                    old_to_new_index[old_index] = new_index
                    new_index += 1

            # Update the edges indices
            edges_idx = [(old_to_new_index[edge[0]], old_to_new_index[edge[1]]) for edge in edges_idx]
            # I don't need to fix the edges labels because I'm removing only clusters without connections
            # for key in edges_labels.keys():
            #     edges_labels[key] = [edges_labels[key][i] for i in range(len(edges_labels[key])) if nodes_features["cl_en"][edges_idx[i][0]] is not None and nodes_features["cl_en"][edges_idx[i][1]] is not None]

            # Remove None values from nodes_features and nodes_sim_features
            for key in nodes_features.keys():
                nodes_features[key] = [value for value in nodes_features[key] if value is not None]
            for key in nodes_sim_features.keys():
                nodes_sim_features[key] = [value for value in nodes_sim_features[key] if value is not None]
            return nodes_features, nodes_sim_features, edges_idx, edges_labels
        
    def pass_simfraction_threshold(self, seed_eta, seed_et, cluster_calo_score ):
        '''
        This functions associates a cluster as true matched if it passes a threshold in simfraction
        '''
        minscore = self.simfraction_thresholds.evaluate(seed_et, abs(seed_eta))
        return cluster_calo_score >= minscore

    def dynamic_window(self,eta, version=3):
        aeta = abs(eta)

        if version == 1:
            if aeta >= 0 and aeta < 0.1:
                deta_up = 0.075
            if aeta >= 0.1 and aeta < 1.3:
                deta_up = 0.0758929 -0.0178571* aeta + 0.0892857*(aeta**2) 
            elif aeta >= 1.3 and aeta < 1.7:
                deta_up = 0.2
            elif aeta >=1.7 and aeta < 1.9:
                deta_up = 0.625 -0.25*aeta
            elif aeta >= 1.9:
                deta_up = 0.15

            if aeta < 2.1: 
                deta_down = -0.075
            elif aeta >= 2.1 and aeta < 2.5:
                deta_down = -0.1875 *aeta + 0.31875
            elif aeta >=2.5:
                deta_down = -0.15

            if aeta < 1.9:
                dphi = 0.6
            elif aeta >= 1.9 and aeta < 2.7:
                dphi = 1.075 -0.25 * aeta
            elif aeta >= 2.7:
                dphi = 0.4

            return deta_up, deta_down, dphi

        elif version == 2:
            if aeta <= 1.5:
                deta_up = (0.1/1.5)*aeta + 0.1
                deta_down = -0.1
            else:
                deta_up = (0.1/1.5)*(aeta-1.5) + 0.2
                deta_down = (-0.1/1.5)*(aeta-1.5) -0.1
            dphi = 0.7 + (-0.1/3)*aeta
            return deta_up, deta_down, dphi


        elif version == 3:
            if aeta <= 1.5:
               deta_up = (0.1/1.5)*aeta + 0.1
               deta_down = -0.1
            else:
                deta_up = (0.1/1.5)*(aeta-1.5) + 0.2
                deta_down = -0.1 + (-0.2/1.5)*(aeta-1.5)
            dphi =  0.7 + (-0.1/3)*aeta
            return deta_up, deta_down, dphi


    def get_windows(self, event, debug=False):
        # Metadata for debugging
        metadata = {
            "n_windows_matched" : 0,
            "n_windows_nomatched" : 0,
            "n_seeds_good":0,
            "n_seeds_bad_calo_position": 0,
            "n_seeds_in_other_window": 0,
        }
        # Branches
        pfCluster_energy = event.pfCluster_energy
        pfCluster_rawEnergy = event.pfCluster_rawEnergy
        pfCluster_eta = event.pfCluster_eta
        pfCluster_phi = event.pfCluster_phi
        pfCluster_ieta = event.pfCluster_ieta
        pfCluster_iphi = event.pfCluster_iphi
        pfCluster_iz = event.pfCluster_iz
        if self.do_pu_sim:
            pfCluster_noise = event.pfCluster_noise
            pfCluster_noise_uncalib  = event.pfCluster_noiseUncalib
            pfCluster_noise_nofrac = event.pfCluster_noiseNoFractions
            pfCluster_noise_uncalib_uncalib = event.pfCluster_noiseUncalibNoFractions
        calo_simenergy = event.caloParticle_simEnergy
        calo_simenergy_goodstatus = event.caloParticle_simEnergyGoodStatus
        calo_genenergy = event.caloParticle_genEnergy
        calo_simeta = event.caloParticle_simEta
        calo_simphi = event.caloParticle_simPhi
        calo_geneta = event.caloParticle_genEta
        calo_genphi = event.caloParticle_genPhi
        calo_simiz = event.caloParticle_simIz
        # calo_geniz = event.caloParticle_genIz
        # calo_isPU = event.caloParticle_isPU
        # calo_isOOTPU = event.caloParticle_isOOTPU
        pfcl_f5_r9 = event.pfCluster_full5x5_r9
        pfcl_f5_sigmaIetaIeta = event.pfCluster_full5x5_sigmaIetaIeta
        pfcl_f5_sigmaIetaIphi = event.pfCluster_full5x5_sigmaIetaIphi
        pfcl_f5_sigmaIphiIphi = event.pfCluster_full5x5_sigmaIphiIphi
        pfcl_f5_swissCross = event.pfCluster_full5x5_swissCross
        pfcl_r9 = event.pfCluster_r9
        pfcl_sigmaIetaIeta = event.pfCluster_sigmaIetaIeta
        pfcl_sigmaIetaIphi = event.pfCluster_sigmaIetaIphi
        pfcl_sigmaIphiIphi = event.pfCluster_sigmaIphiIphi
        pfcl_swissCross = event.pfCluster_swissCross
        pfcl_nxtals = event.pfCluster_nXtals
        pfcl_etaWidth = event.pfCluster_etaWidth
        pfcl_phiWidth = event.pfCluster_phiWidth
        pfclhit_energy = event.pfClusterHit_rechitEnergy
        pfclhit_fraction = event.pfClusterHit_fraction
        pfclhit_ieta = event.pfClusterHit_ieta
        pfclhit_iphi = event.pfClusterHit_iphi
        pfclhit_iz = event.pfClusterHit_iz
        nVtx = event.nVtx
        rho = event.rho
        obsPU = event.obsPU
        truePU = event.truePU
        # pfclhit_eta = event.pfClusterHit_eta
        # pfclhit_phi = event.pfClusterHit_phi

        clusters_scores = getattr(event, "pfCluster_"+self.assoc_strategy)
        # Get Association between pfcluster and calo
        # Sort the clusters for each calo in order of score. 
        # # This is needed to understand which cluster is the seed of the calo
        # Working only on signal caloparticle
        pfcluster_calo_map, pfcluster_calo_score, calo_pfcluster_map = \
                                calo_association.get_calo_association(clusters_scores, sort_calo_cl=True, debug=False, min_sim_fraction=self.cluster_min_fraction)
        # CaloParticle Pileup information
        cluster_signal_simenergy = event.pfCluster_simEnergy_sharedXtals
        if self.do_pu_sim:
            cluster_nXtalsPU = event.pfCluster_simPU_nSharedXtals 
            cluster_PU_simenergy = event.pfCluster_simEnergy_sharedXtalsPU
            cluster_PU_recoenergy = event.pfCluster_recoEnergy_sharedXtalsPU
            total_PU_simenergy = event.caloParticlePU_totEnergy
            
        # #total PU simenergy in all clusters in the event
        # total_PU_simenergy = sum([simPU for cl, simPU in cluster_PU_simenergy.items()])

        if debug:
            print(">>> Cluster_calo map")
            for cluster, calo in pfcluster_calo_map.items():
                if calo == -1: continue
                print("cl: {} | calo: {} (calo Et: {:.2f}, eta {:.2f}, phi {:.2f})| score: {:.4f}, simEnPU: {:.3f}".format(cluster,calo,
                                            calo_simenergy[calo]/cosh(calo_simeta[calo]) ,calo_simeta[calo],calo_simphi[calo],pfcluster_calo_score[cluster],cluster_PU_simenergy[cluster]))
            print("\n>>> Calo_cluster map")
            for calo, clusters in calo_pfcluster_map.items():
                print("calo: {} | clusters: ".format(calo))
                for cl, sc in clusters:
                    print("\t> cl: {}, Et: {:.2f}, eta: {:.2f}, phi:{:.2f}, score: {:.4f}, simEnPU: {:.3f}".format(cl,pfCluster_rawEnergy[cl]/ cosh(pfCluster_eta[cl]), pfCluster_eta[cl],pfCluster_phi[cl], sc,cluster_PU_simenergy[cl]))
            print()

        #Mustache info
        mustacheseed_pfcls = [s for s in event.superCluster_seedIndex]
        mustache_rawEn = event.superCluster_rawEnergy
        mustache_calibEn = event.superCluster_energy
        mustache_eta = event.superCluster_eta
        pfcl_in_mustache = event.superCluster_pfClustersIndex
    
        # map of windows, key=window index
        windows_map = OrderedDict()
        windows_nocalomatched = []
        windows_calomatched = [ ]
        seed_clusters = []
        
        # 1) Look for highest energy cluster (corrected energy)
        clenergies_ordered = sorted([ (ic , et) for ic, et in enumerate(
                                map ( lambda k: k[0]/cosh(k[1]), zip( pfCluster_energy, pfCluster_eta) )
                                )], key=itemgetter(1), reverse=True)

        if debug: print(">> Windows formation")

        # First we loop over the classical "windows" to store the edges and labels.
        Ncls = len(pfCluster_energy)
        nodes_features = {
            "cl_en": [0.]*Ncls,
            "cl_et": [0.]*Ncls,
            "cl_en_calib": [0.]*Ncls,
            "cl_et_calib": [0.]*Ncls,
            "cl_eta": [0.]*Ncls,
            "cl_phi": [0.]*Ncls,
            "cl_ieta": [0.]*Ncls,
            "cl_iphi": [0.]*Ncls,
            "cl_iz": [0.]*Ncls,
            "cl_f5_r9": [0.]*Ncls,
            "cl_f5_sigmaIetaIeta": [0.]*Ncls,
            "cl_f5_sigmaIetaIphi": [0.]*Ncls,
            "cl_f5_sigmaIphiIphi": [0.]*Ncls,
            "cl_f5_swissCross": [0.]*Ncls,
            "cl_r9": [0.]*Ncls,
            "cl_sigmaIetaIeta": [0.]*Ncls,
            "cl_sigmaIetaIphi": [0.]*Ncls,
            "cl_sigmaIphiIphi": [0.]*Ncls,
            "cl_etaWidth": [0.]*Ncls,
            "cl_phiWidth": [0.]*Ncls,
            "cl_nxtals": [0.]*Ncls,
        }
        nodes_sim_features = {
            "is_calo_matched": [0]*Ncls,
            "is_calo_seed":  [0]*Ncls,
            "calo_index": [-1]*Ncls, # default -1 for no matching
            "calo_score": [-1]*Ncls,
            #"calo_simen_sig": [0.]*Ncls,
            "en_true_sim": [0.]*Ncls,
            "et_true_sim": [0.]*Ncls,
            "en_true_gen": [0.]*Ncls,
            "et_true_gen": [0.]*Ncls,
        }
        global_features = {}
        edges_idx = []
        edges_labels = {
            "in_scluster": [],
            #"calo_score": []  #score of the cluster wrt of the linked cluster-associated calo
            "is_calo_matched_same": [], # flag True is the cluster is associated to the same calo of the linked cluster
            "is_calo_seed_same": [] # flag True if the cluster is the real seed of the calo associated with the linked cluster
        }
        noncalomatched_clusters = []
        
        # Now iterate over clusters in order of energies
        for icl, clenergy_T in clenergies_ordered:
            #print(icl, clenergy_T)

            # No seeds with Et< min_et_seed GeV
            if clenergy_T < self.min_et_seed: continue
            # No seed with Et> max_et_seed GeV
            if clenergy_T > self.max_et_seed: continue

            cl_eta = pfCluster_eta[icl]
            cl_phi = pfCluster_phi[icl]
            cl_ieta = pfCluster_ieta[icl]
            cl_iphi = pfCluster_iphi[icl]
            cl_iz =  pfCluster_iz[icl]

            is_in_window = False
            # Check if it is already in one windows
            # - Check if the seed is associated with a calo in the window: calomatched 
            # - Check if the seed is the main cluster of the calo:  caloseed
            # It is required to have seed_min_fraction% of the calo energy and to be "in the window" of the seed
            if pfcluster_calo_map[icl] !=-1 and pfcluster_calo_score[icl] > self.seed_min_fraction:
                # ID of the associated caloparticle
                caloid = pfcluster_calo_map[icl] # --> caloparticle of the seed
                if self.do_pu_sim:
                    # Do not check PU fraction on the seed
                    PU_simenfrac = cluster_PU_simenergy[icl] / cluster_signal_simenergy[icl][caloid]
                #Check if the caloparticle is in the same window with GEN info
                if in_window(calo_geneta[caloid],calo_genphi[caloid],calo_simiz[caloid], cl_eta, cl_phi, cl_iz, 
                                                *self.dynamic_window(cl_eta)):
                    calomatched = caloid
                    metadata["n_seeds_good"] +=1
                    # Now check if the seed cluster is the main cluster of the calo
                    if calo_pfcluster_map[caloid][0][0] == icl:
                        caloseed = True
                    else:
                        caloseed =False
                    if debug: 
                        print("Seed-to-calo: cluster: {}, calo: {}, seed_eta: {:.3f}, calo_genEta : {:.3f}, seed_score: {:.5f}, is caloseed: {}".format(
                                            icl,caloid,cl_eta,calo_geneta[caloid], pfcluster_calo_score[icl], caloseed))
                else:
                    metadata["n_seeds_bad_calo_position"] +=1
                    calomatched = -1
                    caloseed = False
                    if debug: 
                        print("Seed-to-calo [Failed window cut]: cluster: {}, calo: {}, seed_eta: {:.3f}, calo_eta : {:.3f}, seed_score: {:.5f}, is caloseed: {}".format(
                                            icl, caloid,cl_eta,calo_geneta[caloid], pfcluster_calo_score[icl], caloseed))

            else:
                metadata["n_windows_nomatched"] += 1
                calomatched = -1
                caloseed = False
                PU_simenfrac = -1.

            # Save the cluster in the list of associated clusters
            seed_clusters.append(icl)
            # Check if it is a mustache seed
            if icl in mustacheseed_pfcls:
                mustache_seed_index = mustacheseed_pfcls.index(icl)
            else:
                mustache_seed_index = -1

            # Let's create  new window:
            if calomatched == -1:
                noncalomatched_clusters.append(icl)

            # saving features
            nodes_sim_features["is_calo_matched"][icl] = 1 if calomatched != -1 else 0
            nodes_sim_features["is_calo_seed"][icl] = 1 if caloseed else 0
            nodes_sim_features["calo_index"][icl] = calomatched if calomatched != -1 else -1
            nodes_sim_features["calo_score"][icl] = pfcluster_calo_score[icl] if pfcluster_calo_map[icl] != -1 else -1
            nodes_sim_features["en_true_sim"][icl] = calo_simenergy_goodstatus[calomatched] if calomatched!=-1 else 0
            nodes_sim_features["et_true_sim"][icl] = calo_simenergy_goodstatus[calomatched]/cosh(calo_simeta[calomatched]) if calomatched!=-1 else 0
            nodes_sim_features["en_true_gen"][icl] = calo_genenergy[calomatched] if calomatched!=-1 else 0
            nodes_sim_features["et_true_gen"][icl] = calo_genenergy[calomatched]/cosh(calo_geneta[calomatched]) if calomatched!=-1 else 0
            global_features["nVtx"] = nVtx
            global_features["rho"] = rho
            global_features["obsPU"] = obsPU
            global_features["truePU"] = truePU
            global_features["event_tot_simen_PU"] = total_PU_simenergy if self.do_pu_sim else 0.


        ####################################
        ## Now loop on clusters

        # All the clusters will go in all the windows
        if debug: print(">> Associate clusters...")
        # Now that all the windows have been created let's add all the cluster
        for icl, clenergy_T in clenergies_ordered:
            
            cl_eta = pfCluster_eta[icl]
            cl_phi = pfCluster_phi[icl]
            cl_ieta = pfCluster_ieta[icl]
            cl_iphi = pfCluster_iphi[icl]
            cl_iz = pfCluster_iz[icl]
            cl_rawen = pfCluster_rawEnergy[icl]

            # Storing all the kin info of all the clusters before looking at connections
            nodes_features["cl_en"][icl] = pfCluster_rawEnergy[icl]
            nodes_features["cl_et"][icl] = pfCluster_rawEnergy[icl]/cosh(pfCluster_eta[icl])
            nodes_features["cl_en_calib"][icl] = pfCluster_energy[icl]
            nodes_features["cl_et_calib"][icl] = pfCluster_energy[icl]/cosh(pfCluster_eta[icl])
            nodes_features["cl_eta"][icl] = cl_eta
            nodes_features["cl_phi"][icl] = cl_phi
            nodes_features["cl_ieta"][icl] = cl_ieta
            nodes_features["cl_iphi"][icl] = cl_iphi
            nodes_features["cl_iz"][icl] = cl_iz
            nodes_features["cl_f5_r9"][icl] = pfcl_f5_r9[icl]
            nodes_features["cl_f5_sigmaIetaIeta"][icl] = pfcl_f5_sigmaIetaIeta[icl]
            nodes_features["cl_f5_sigmaIetaIphi"][icl] = pfcl_f5_sigmaIetaIphi[icl]
            nodes_features["cl_f5_sigmaIphiIphi"][icl] = pfcl_f5_sigmaIphiIphi[icl]
            nodes_features["cl_f5_swissCross"][icl] = pfcl_f5_swissCross[icl]
            nodes_features["cl_r9"][icl] = pfcl_r9[icl]
            nodes_features["cl_sigmaIetaIeta"][icl] = pfcl_sigmaIetaIeta[icl]
            nodes_features["cl_sigmaIetaIphi"][icl] = pfcl_sigmaIetaIphi[icl]
            nodes_features["cl_sigmaIphiIphi"][icl] = pfcl_sigmaIphiIphi[icl]
            nodes_features["cl_etaWidth"][icl] = pfcl_etaWidth[icl]
            nodes_features["cl_phiWidth"][icl] = pfcl_phiWidth[icl]
            nodes_features["cl_nxtals"][icl] = pfcl_nxtals[icl]
                      
            for iseed in seed_clusters:

                if icl == iseed:
                    continue
                
                seed_eta = pfCluster_eta[iseed]
                seed_phi = pfCluster_phi[iseed]
                seed_iz = pfCluster_iz[iseed]
                et_seed = pfCluster_energy[iseed]/cosh(pfCluster_eta[iseed])
                seed_calo_index = nodes_sim_features["calo_index"][iseed]
                
                isin, (etaw, phiw) = in_window(
                                                seed_eta, seed_phi, seed_iz,
                                                 cl_eta, cl_phi, cl_iz,
                                                *self.dynamic_window(seed_eta))
                if isin:
                    # Looking at the window of the other cluster
                    # First of all check is the cluster is geometrically in the mustache of the seed
                    in_geom_mustache = is_in_geom_mustache( seed_eta, seed_phi,
                                                           cl_eta, cl_phi, cl_rawen)
                    # If the window is not associated to a calo then in_scluster is always false for the cluster
                    if nodes_sim_features["is_calo_matched"][iseed] == 0:
                        is_calo_matched = False   
                        pass_simfrac_thres = False
                        is_calo_seed = False
                        PU_simenfrac = -1.
                    else: 
                        # We have to check the calo_matching using simfraction threshold
                        # Check if the cluster is associated to the SAME calo as the seed
                        is_calo_matched =  pfcluster_calo_map[icl] == seed_calo_index  # we know at this point it is not -1
                        # Check the case for a cluster matched to a caloparticle that is different from the caloparticle of the seed
                        is_calo_matched_different_calo =  (is_calo_matched == False) and (pfcluster_calo_map[icl] != -1)
                        # If the cluster is associated to the SAME CALO of the seed 
                        # the simfraction threshold by seed eta/et is used
                        if is_calo_matched: 
                            # Check the fraction of sim energy and PU energy 
                            # simenergy signal == linked to the caloparticle of the SEED
                            if self.do_pu_sim:
                                PU_simenfrac = cluster_PU_simenergy[icl] / cluster_signal_simenergy[icl][window["calo_index"]]
                            else:
                                PU_simenfrac = 0
                            # First of all check the PU sim energy limit
                            if PU_simenfrac < self.simenergy_pu_limit:
                                #associate the cluster to the caloparticle with simfraction optimized thresholds 
                                pass_simfrac_thres = self.pass_simfraction_threshold(seed_eta, 
                                                                                     et_seed,
                                                                                     pfcluster_calo_score[icl] )
                                # Check if the cluster is the main cluster of the calo associated to the seed
                                if calo_pfcluster_map[seed_calo_index][0][0] == icl:
                                    is_calo_seed = True
                                else:
                                    is_calo_seed = False
                            else:
                                if debug: print("Cluster {} do not pass PU simenergy cut {:.3f}".format(icl, PU_simenfrac))
                                pass_simfrac_thres = False   
                                is_calo_seed = False

                        elif is_calo_matched_different_calo:
                            if calo_pfcluster_map[pfcluster_calo_map[icl]][0][0] == icl:
                                is_calo_seed = True
                                othercalo = pfcluster_calo_map[icl]
                                pass_simfrac_thres = False
                            else:
                                is_calo_seed = False
                                othercalo = -1
                                pass_simfrac_thres = False
                    
                        else:
                            # if the cluster is not associated to a caloparticle
                            # or it is associated to a calo different from the seed
                            # do not associate it
                            pass_simfrac_thres = False   
                            is_calo_seed = False
                            PU_simenfrac = -1.            
                            
                    # We store the edge and its properties in the graph
                    edges_idx.append((icl, iseed))
                    edges_labels["in_scluster"].append(1 if pass_simfrac_thres or (iseed == icl) else 0)
                    edges_labels["is_calo_matched_same"].append(1 if is_calo_matched else 0)
                    edges_labels["is_calo_seed_same"].append(1 if is_calo_seed else 0) #is the caloseed of the calopart matched to the iseed


        ###############
        # We want to keep only a max number of non-calomatched clusters with no connections
        noncalomatched_no_connections = []

        # Check if there are noncalomatched clusters without connections
        for icl in noncalomatched_clusters:
            has_connection = any(edge[0] == icl or edge[1] == icl for edge in edges_idx)
            if not has_connection:
                noncalomatched_no_connections.append(icl)

        # I want to keep only a max number of noncalomatched clusters
        if len(noncalomatched_no_connections) > 0:
            random.shuffle(noncalomatched_no_connections)
            noncalomatched_clusters_toremove = noncalomatched_no_connections[min(self.nocalomatchedNmax,len(noncalomatched_no_connections)):]
        else:
            noncalomatched_clusters_toremove = []

        (nodes_features, nodes_sim_features,
         edges_idx, edges_labels) =  self.remove_clusters(noncalomatched_clusters_toremove, nodes_features,
                        nodes_sim_features, edges_idx, edges_labels)

        # Now I also want to identify cluster above the energy thresholds with no connections
        clusters_to_remove = []
        for icl in range(len(nodes_features["cl_et"])):
            if nodes_features["cl_et"][icl] is not None:
                if nodes_features["cl_et"][icl] > self.max_et_isolated_cl:
                    has_connection = any(edge[0] == icl or edge[1] == icl for edge in edges_idx)
                    if not has_connection:
                        clusters_to_remove.append(icl)

        #print("Clusters to remove: ", clusters_to_remove)
        (nodes_features, nodes_sim_features,
         edges_idx, edges_labels) =  self.remove_clusters(clusters_to_remove, nodes_features,
                        nodes_sim_features, edges_idx, edges_labels)

        ############
        # print(seed_clusters)
        # print(edges_idx)
        # print(edges_labels)
        # We need to split the output in 3 by iz of the cluster
        # 1) iz = 0  0
        # 2) iz = 1  1
        # 3) iz = -1 2
        output_data = {
            "nodes_features": nodes_features,
            "nodes_sim_features": nodes_sim_features,
            "global_features": global_features,
            "edges_idx": edges_idx,
            "edges_labels": edges_labels
        }
        
        output_data_split = [ {
            "nodes_features": { key: [ ] for key in nodes_features.keys() },
            "nodes_sim_features": { key: [ ] for key in nodes_sim_features.keys() },
            "global_features": global_features,
            "edges_idx": [],
            "edges_labels": { key: [] for key in edges_labels.keys()}
        } for i in range(3)]
        

        idx_split =[[],[],[]]
        curr_idx_split = [-1,-1,-1]
        new_index_ref = []

        for icl in range(len(output_data["nodes_features"]["cl_en"])):
            if output_data["nodes_features"]["cl_iz"][icl] == 0:
                iz_idx = 0
            elif output_data["nodes_features"]["cl_iz"][icl] == 1:
                iz_idx = 1
            elif output_data["nodes_features"]["cl_iz"][icl] == -1:
                iz_idx = 2
               
            idx_split[iz_idx].append(icl)
            curr_idx_split[iz_idx] += 1
            new_index_ref.append((iz_idx, curr_idx_split[iz_idx]))
            
            for key in output_data["nodes_features"].keys():
                output_data_split[iz_idx]["nodes_features"][key].append(output_data["nodes_features"][key][icl])
            for key in output_data["nodes_sim_features"].keys():
                output_data_split[iz_idx]["nodes_sim_features"][key].append(output_data["nodes_sim_features"][key][icl])

        for iedge, edge in enumerate(output_data["edges_idx"]):
            idx1 = new_index_ref[edge[0]]
            idx2 = new_index_ref[edge[1]]
            if idx1[0] == idx2[0]:
                output_data_split[idx1[0]]["edges_idx"].append((idx1[1], idx2[1]))
                for key in output_data["edges_labels"].keys():
                    output_data_split[idx1[0]]["edges_labels"][key].append(output_data["edges_labels"][key][iedge])

                            
        # if debug: print(output_data)
        output_data_str = [ json.dumps(out) for out in output_data_split]
        return output_data_str, metadata

