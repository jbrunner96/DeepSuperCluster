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
                 min_et_seed=1., assoc_strategy="sim_fraction", overlapping_window=False,  nocalowNmax=0, do_pu_sim=False):
        self.seed_min_fraction = seed_min_fraction
        self.cluster_min_fraction = cl_min_fraction
        self.simfraction_thresholds = correctionlib.CorrectionSet.from_file(simfraction_thresholds)["simfraction_thres"]
        self.simenergy_pu_limit = simenergy_pu_limit
        self.min_et_seed=min_et_seed
        self.assoc_strategy = assoc_strategy
        self.overlapping_window = overlapping_window
        self.nocalowNmax = nocalowNmax
        self.do_pu_sim = do_pu_sim

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
        # Now iterate over clusters in order of energies
        for icl, clenergy_T in clenergies_ordered:
            #print(icl, clenergy_T)

            # No seeds with Et< min_et_seed GeV
            if clenergy_T < self.min_et_seed: continue

            cl_eta = pfCluster_eta[icl]
            cl_phi = pfCluster_phi[icl]
            cl_ieta = pfCluster_ieta[icl]
            cl_iphi = pfCluster_iphi[icl]
            cl_iz =  pfCluster_iz[icl]

            is_in_window = False
            # Check if it is already in one windows
            for window in windows_map.values():
                is_in_this_window, (etaw, phiw) = in_window(*window["seed"], cl_eta, cl_phi, cl_iz, 
                                                    *self.dynamic_window(window["seed"][0])) 
                if is_in_this_window:
                    is_in_window = True
                    if debug: print("Cluster {} already in window {}! skipping window".format(icl, window["window_index"]))
                    break

                
            if not self.overlapping_window and is_in_window:
                # If we are in non-overlapping mode 
                # Create new window ONLY IF THE CLUSTER IS NOT ALREADY IN ANOTHER WINDOW
                continue
            else:
                # - Check if the seed simFraction with the signal calo is at least seed_min_fraction
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


                # Create a unique index
                windex = "".join([ random.choice(string.ascii_lowercase) for _ in range(9)])
                # Let's create  new window:
                new_window = {
                    "window_index": windex,
                    "seed_index": icl,
                    "seed": (cl_eta, cl_phi, cl_iz),
                    # The seed of the window is associated with a caloparticle in the window
                    "is_seed_calo_matched": calomatched != -1,
                    # index of the associated caloparticle
                    "calo_index": calomatched,
                    # The seed is the cluster associated with the particle with the largest fraction
                    "is_seed_calo_seed": caloseed,
                    # Mustache info
                    "is_seed_mustache_matched": mustache_seed_index != -1,
                    "mustache_seed_index": mustache_seed_index,
                    
                    # Score of the seed cluster
                    "seed_score": pfcluster_calo_score[icl],
                    "seed_simen_sig": cluster_signal_simenergy[icl][calomatched] if calomatched!=-1 else 0.,
                    "seed_simen_PU":  cluster_PU_simenergy[icl] if self.do_pu_sim else 0.,
                    "seed_recoen_PU":  cluster_PU_recoenergy[icl] if self.do_pu_sim else 0.,
                    "seed_PUfrac" : PU_simenfrac if self.do_pu_sim else 0.,

                    "seed_eta": cl_eta,
                    "seed_phi": cl_phi, 
                    "seed_iz": cl_iz,
                    "seed_ieta": cl_ieta,
                    "seed_iphi": cl_iphi, 

                    # Sim position
                    "sim_true_eta" : calo_simeta[calomatched] if calomatched!=-1 else 0, 
                    "sim_true_phi":  calo_simphi[calomatched] if calomatched!=-1 else 0, 
                    "gen_true_eta" : calo_geneta[calomatched] if calomatched!=-1 else 0, 
                    "gen_true_phi":  calo_genphi[calomatched] if calomatched!=-1 else 0, 

                    # Energy of the seed
                    "en_seed": pfCluster_rawEnergy[icl],
                    "et_seed": pfCluster_rawEnergy[icl] / cosh(cl_eta),
                    "en_seed_calib": pfCluster_energy[icl],
                    "et_seed_calib": pfCluster_energy[icl] / cosh(cl_eta),

                    "en_seed_log": log(1 + pfCluster_rawEnergy[icl]),
                    "et_seed_log": log(1 + pfCluster_rawEnergy[icl] / cosh(cl_eta)),
                    "en_seed_calib_log": log(1 + pfCluster_energy[icl]),
                    "et_seed_calib_log": log(1 + pfCluster_energy[icl] / cosh(cl_eta)),

                    # Sim energy and Gen Enerugy of the caloparticle
                    "en_true_sim": calo_simenergy[calomatched] if calomatched!=-1 else 0, 
                    "et_true_sim": calo_simenergy[calomatched]/cosh(calo_geneta[calomatched]) if calomatched!=-1 else 0, 
                    "en_true_gen": calo_genenergy[calomatched] if calomatched!=-1 else 0, 
                    "et_true_gen": calo_genenergy[calomatched]/cosh(calo_geneta[calomatched]) if calomatched!=-1 else 0,
                    "en_true_sim_good": calo_simenergy_goodstatus[calomatched] if calomatched!=-1 else 0, 
                    "et_true_sim_good": calo_simenergy_goodstatus[calomatched]/cosh(calo_geneta[calomatched]) if calomatched!=-1 else 0,
                    
                    # Energy of the mustache if present. Raw and regressed
                    "en_mustache_raw": mustache_rawEn[mustache_seed_index] if mustache_seed_index!=-1 else 0, 
                    "et_mustache_raw": mustache_rawEn[mustache_seed_index]/cosh(mustache_eta[mustache_seed_index]) if mustache_seed_index!=-1 else 0, 
                    "en_mustache_calib": mustache_calibEn[mustache_seed_index]  if mustache_seed_index!=-1 else 0, 
                    "et_mustache_calib": mustache_calibEn[mustache_seed_index]/cosh(mustache_eta[mustache_seed_index]) if mustache_seed_index!=-1 else 0,

                    # PU information
                    "nVtx": nVtx, 
                    "rho": rho,
                    "obsPU": obsPU, 
                    "truePU": truePU,
                    "event_tot_simen_PU": total_PU_simenergy if self.do_pu_sim else 0.,

                    "seed_f5_r9": pfcl_f5_r9[icl],
                    "seed_f5_sigmaIetaIeta" : pfcl_f5_sigmaIetaIeta[icl],
                    "seed_f5_sigmaIetaIphi" : pfcl_f5_sigmaIetaIphi[icl],
                    "seed_f5_sigmaIphiIphi" : pfcl_f5_sigmaIphiIphi[icl],
                    "seed_f5_swissCross" : pfcl_f5_swissCross[icl],
                    "seed_r9": pfcl_r9[icl],
                    "seed_sigmaIetaIeta" : pfcl_sigmaIetaIeta[icl],
                    "seed_sigmaIetaIphi" : pfcl_sigmaIetaIphi[icl],
                    "seed_sigmaIphiIphi" : pfcl_sigmaIphiIphi[icl],
                    "seed_swissCross" : pfcl_swissCross[icl],

                    "seed_etaWidth" : pfcl_etaWidth[icl],
                    "seed_phiWidth" : pfcl_phiWidth[icl],
                    "seed_nxtals" : pfcl_nxtals[icl],
                    "seed_hits" : get_cluster_hits(pfclhit_ieta[icl], pfclhit_iphi[icl],pfclhit_iz[icl], pfclhit_energy[icl], pfclhit_fraction[icl]), 
                
                    "clusters": [],

                }
                # Save the window
                windows_map[windex] = new_window
                if calomatched == -1:  
                    windows_nocalomatched.append(windex)
                else:
                    # Save also the window index
                    windows_calomatched.append(windex)

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

            # print("Checking cluster: ",icl)
            # Fill all the windows
            for window in windows_map.values():
                isin, (etaw, phiw) = in_window(*window["seed"], cl_eta, cl_phi, cl_iz,
                                                *self.dynamic_window(window["seed"][0]))
                #print("\t w: ", window['window_index'], isin)
                if isin:
                    # First of all check is the cluster is geometrically in the mustache of the seed
                    in_geom_mustache = is_in_geom_mustache(window["seed_eta"], 
                                                window["seed_phi"], cl_eta, cl_phi, cl_rawen)
                    # If the window is not associated to a calo then in_scluster is always false for the cluster
                    if not window["is_seed_calo_matched"]:
                        is_calo_matched = False   
                        pass_simfrac_thres = False
                        is_calo_seed = False
                        PU_simenfrac = -1.
                    else: 
                        # We have to check the calo_matching using simfraction threshold
                        # Check if the cluster is associated to the SAME calo as the seed
                        is_calo_matched =  pfcluster_calo_map[icl] == window["calo_index"]  # we know at this point it is not -1
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
                                pass_simfrac_thres = self.pass_simfraction_threshold(window["seed_eta"], 
                                                    window["et_seed"], pfcluster_calo_score[icl] )
                                # Check if the cluster is the main cluster of the calo associated to the seed
                                if calo_pfcluster_map[window["calo_index"]][0][0] == icl:
                                    is_calo_seed = True
                                else:
                                    is_calo_seed =False
                            else:
                                if debug: print("Cluster {} do not pass PU simenergy cut {:.3f}".format(icl, PU_simenfrac))
                                pass_simfrac_thres = False   
                                is_calo_seed = False

                        elif is_calo_matched_different_calo:
                            if calo_pfcluster_map[pfcluster_calo_map[icl]][0][0] == icl:
                                is_calo_seed = True
                                othercalo = pfcluster_calo_map[icl]
                            else:
                                is_calo_seed = False
                                othercalo = -1
                    
                        else:
                            # if the cluster is not associated to a caloparticle
                            # or it is associated to a calo different from the seed
                            # do not associate it
                            pass_simfrac_thres = False   
                            is_calo_seed = False
                            PU_simenfrac = -1.            
                            
                    # check if the cluster is inside the same mustache
                    if window["mustache_seed_index"] != -1:
                        in_mustache = icl in pfcl_in_mustache[window["mustache_seed_index"]]
                    else:
                        in_mustache = False
                
                    cevent = {  
                        # "window_index": window["window_index"],
                        "cl_index": icl,
                        # Check if it is the seed
                        "is_seed": window["seed_index"] == icl,
                        # True if the cluster geometrically is in the mustache of the seed
                        "in_geom_mustache" : in_geom_mustache,
                        # True if the seed has a calo and the cluster is associated to the same calo
                        "is_calo_matched": is_calo_matched,
                        "is_calo_matched_diffcalo": is_calo_matched_different_calo,
                        "other_calo_index": othercalo if is_calo_matched_different_calo else -1,
                        # True if the cluster is the main cluster of the calo associated with the seed
                        "is_calo_seed": is_calo_seed,
                        # is_calo_matched & (sim fraction optimized threshold) || cl it is the seed of the window 
                        "in_scluster": pass_simfrac_thres or (window["seed_index"] == icl) ,
                        # True if the cluster is associated with the same (legacy) mustache as the seed
                        #"in_mustache" : in_mustache, # saving only the in_geom_mustache, because the SC in the dumper can be one of the two algos
                        # Score of association with the caloparticle of the seed, if present
                        "calo_score": pfcluster_calo_score[icl],
                        # Simenergy of the signal and PU in the cluster
                        "calo_simen_sig": cluster_signal_simenergy[icl][window["calo_index"]] if is_calo_matched else 0.,
                        "calo_simen_PU":  cluster_PU_simenergy[icl] if self.do_pu_sim else 0.,
                        "calo_recoen_PU": cluster_PU_recoenergy[icl] if self.do_pu_sim else 0.,
                        "calo_nxtals_PU": cluster_nXtalsPU[icl] if self.do_pu_sim else 0.,
                        "cluster_PUfrac": PU_simenfrac if self.do_pu_sim else 0.,

                        "cluster_ieta" : cl_ieta,
                        "cluster_iphi" : cl_iphi,
                        "cluster_eta" : cl_eta,
                        "cluster_phi" : cl_phi,
                        "cluster_dphi":phiw ,
                        "cluster_iz" : cl_iz,
                        "en_cluster": pfCluster_rawEnergy[icl],
                        "et_cluster": pfCluster_rawEnergy[icl] / cosh(cl_eta),
                        "en_cluster_calib": pfCluster_energy[icl],
                        "et_cluster_calib": pfCluster_energy[icl] /cosh(cl_eta),

                        "en_cluster_log": log(1 + pfCluster_rawEnergy[icl]),
                        "et_cluster_log": log(1 + pfCluster_rawEnergy[icl] / cosh(cl_eta)),
                        "en_cluster_calib_log": log(1 + pfCluster_energy[icl]),
                        "et_cluster_calib_log": log(1 + pfCluster_energy[icl] /cosh(cl_eta)),
                        
                        "noise_en" : pfCluster_noise[icl] if self.do_pu_sim else 0.,
                        "noise_en_uncal": pfCluster_noise_uncalib[icl] if self.do_pu_sim else 0.,
                        "noise_en_nofrac": pfCluster_noise_nofrac[icl] if self.do_pu_sim else 0.,
                        "noise_en_uncal_nofrac": pfCluster_noise_uncalib_uncalib[icl] if self.do_pu_sim else 0.,
                        
                        # Shower shape variables
                        "cl_f5_r9": pfcl_f5_r9[icl],
                        "cl_f5_sigmaIetaIeta" : pfcl_f5_sigmaIetaIeta[icl],
                        "cl_f5_sigmaIetaIphi" : pfcl_f5_sigmaIetaIphi[icl],
                        "cl_f5_sigmaIphiIphi" : pfcl_f5_sigmaIphiIphi[icl],
                        "cl_f5_swissCross" : pfcl_f5_swissCross[icl],
                        "cl_r9": pfcl_r9[icl],
                        "cl_sigmaIetaIeta" : pfcl_sigmaIetaIeta[icl],
                        "cl_sigmaIetaIphi" : pfcl_sigmaIetaIphi[icl],
                        "cl_sigmaIphiIphi" : pfcl_sigmaIphiIphi[icl],
                        "cl_swissCross" : pfcl_swissCross[icl],

                        "cl_etaWidth" : pfcl_etaWidth[icl],
                        "cl_phiWidth" : pfcl_phiWidth[icl],
                        "cl_nxtals" : pfcl_nxtals[icl],

                        "cl_hits":  get_cluster_hits(pfclhit_ieta[icl], pfclhit_iphi[icl],pfclhit_iz[icl], pfclhit_energy[icl], pfclhit_fraction[icl])
                    }
                    if window["seed_eta"] > 0:
                        cevent["cluster_deta"] = cl_eta - window["seed_eta"]
                    else:
                        cevent["cluster_deta"] = window["seed_eta"] - cl_eta
                    # Delta energy with the seed
                    cevent["cluster_den_seed"] = window["en_seed"] - cevent["en_cluster"]
                    cevent["cluster_det_seed"] = window["et_seed"] - cevent["et_cluster"]
                    cevent["cluster_den_seed_log"] = np.log(cevent["cluster_den_seed"] + 1)
                    cevent["cluster_det_seed_log"] = np.log(cevent["cluster_det_seed"] + 1)
                    
                    # Save the cluster in the window
                    window["clusters"].append(cevent)
                    # In this script save all the clusters in all the windows
                    # Uncomment the next line if instead you want to
                    # save only the cluster in the first window encountered by Et
                    #break

        ###############################
        #### Now that all the clusters have been put in all the windows
        ###  Add some global data for each window
        
        for window in windows_map.values():
            calo_index = window["calo_index"]
            # Check the type of events
            # - Number of pfcluster associated, 
            # - deltaR of the farthest cluster
            # - Energy of the pfclusters
            if calo_index != -1:
                # Get number of associated clusters
                assoc_clusters = list(filter(lambda cl: cl["in_scluster"], window["clusters"]))
                window["nclusters_insc"] = len(assoc_clusters)
                window["max_en_cluster_insc"] = max( [cl["en_cluster"] for cl in assoc_clusters ] )
                window["max_deta_cluster_insc"] = max( [abs(cl["cluster_deta"]) for cl in assoc_clusters] )
                window["max_dphi_cluster_insc"] = max( [abs(cl["cluster_dphi"]) for cl in assoc_clusters])
            else:
                window["nclusters_insc"] = 0
                window["max_en_cluster_insc"] = -1
                window["max_deta_cluster_insc"] = -1
                window["max_dphi_cluster_insc"] = -1
            #General info for all the windows
            #max variable
            window["ncls"] = len(window["clusters"])
            window["max_en_cluster"] = max( cl["en_cluster"] for cl in window["clusters"])
            window["max_et_cluster"] = max( cl["et_cluster"] for cl in window["clusters"])
            window["max_deta_cluster"] = max( abs(cl["cluster_deta"]) for cl in window["clusters"])
            window["max_dphi_cluster"] = max( abs(cl["cluster_dphi"]) for cl in window["clusters"])
            window["max_den_cluster"] = max( cl["cluster_den_seed"] for cl in window["clusters"])
            window["max_det_cluster"] = max( cl["cluster_det_seed"] for cl in window["clusters"])
            #min variables
            window["min_en_cluster"] = min( cl["en_cluster"] for cl in window["clusters"])
            window["min_et_cluster"] = min( cl["et_cluster"] for cl in window["clusters"])
            window["min_deta_cluster"] = min( abs(cl["cluster_deta"]) for cl in window["clusters"])
            window["min_dphi_cluster"] = min( abs(cl["cluster_dphi"]) for cl in window["clusters"])
            window["min_den_cluster"] = min( cl["cluster_den_seed"] for cl in window["clusters"])
            window["min_det_cluster"] = min( cl["cluster_det_seed"] for cl in window["clusters"])
            #mean variabes
            window["mean_en_cluster"] = np.mean( list(cl["en_cluster"] for cl in window["clusters"]))
            window["mean_et_cluster"] = np.mean( list(cl["et_cluster"] for cl in window["clusters"]))
            window["mean_deta_cluster"] = np.mean( list(abs(cl["cluster_deta"]) for cl in window["clusters"]))
            window["mean_dphi_cluster"] = np.mean( list(abs(cl["cluster_dphi"]) for cl in window["clusters"]))
            window["mean_den_cluster"] = np.mean( list(cl["cluster_den_seed"] for cl in window["clusters"]))
            window["mean_det_cluster"] = np.mean( list(cl["cluster_det_seed"] for cl in window["clusters"]))
            # Compute total simEnergy of the signal and PU in the window    
            # Take only the calo of the window
            total_PU_simenergy_inwindow = 0. 
            total_PU_recoenergy_inwindow = 0.
            total_sig_simenergy_inwindow = 0.
            for cl in window["clusters"]:
                total_PU_simenergy_inwindow  +=  cl["calo_simen_PU"]
                total_PU_recoenergy_inwindow  +=  cl["calo_recoen_PU"]
                total_sig_simenergy_inwindow += cl["calo_simen_sig"]
            # Saving window totals to differentiate from total in the event
            window["wtot_simen_PU"] = total_PU_simenergy_inwindow
            window["wtot_recoen_PU"] = total_PU_recoenergy_inwindow
            window["wtot_simen_sig"] = total_sig_simenergy_inwindow
        
        if debug:
            print("ALL windows")
            print("N windows:", len(windows_map))
            calo_match = []
            nocalo_match = []
            for w in windows_map.values():
                if w["is_seed_calo_matched"]: calo_match.append(w)
                else: nocalo_match.append(w)
            print("Calo-matched windows: ")
            for w in calo_match:
                print(">> Window: ", w["window_index"], "  Calo Matched: ",  w["is_seed_calo_matched"]," Seed calo-seed: ",w['is_seed_calo_seed'])
                print("\t Calo: {}, Eta:{:.3f}, Phi:{:.3f}, Et:{:.3f}".format(w["calo_index"],
                                        calo_simeta[w["calo_index"]],calo_simphi[w["calo_index"]], w["et_true_sim"]))
                print("\t Seed: {}, Eta:{:.3f}, Phi:{:.3f}, Iz: {:.3f}, Et:{:.3f}".format( w['seed_index'], w["seed_eta"], w["seed_phi"],w["seed_iz"], w["et_seed"]))
                print("\t Total simEnergy signal: {:.3f}, PU: {:.3f}".format(w["wtot_simen_sig"],w["wtot_simen_PU"]))
                print("\t Clusters: ", [cl['cl_index'] for cl in w['clusters']])
                
            print("Not Calo-matched windows: ")
            for w in nocalo_match:
                print(">> Window: ", w["window_index"], "  Calo Matched: ",  w["is_seed_calo_matched"])
                print("\t Seed: {}, Eta:{:.3f}, Phi:{:.3f}, Iz: {:.3f}, Et:{:.3f}".format( w['seed_index'], w["seed_eta"], w["seed_phi"],w["seed_iz"], w["et_seed"]))
                print("\t Total simEnergy signal: {:.3f}, PU: {:.3f}".format(w["wtot_simen_sig"],w["wtot_simen_PU"]))
                print("\t Clusters: ", [cl['cl_index'] for cl in w['clusters']])
                
            print(">>> TOT windows calomatched: ", len(calo_match))
            print(">>> Tot PU simEnergy in the event: ", total_PU_simenergy)

        # Check if there are more than one windows associated with the same caloparticle
        # windows_calomatched = []
        # for calo, ws in calo_windows.items():
        #     # Take only the first windows, by construction the most energetic one
        #     windows_calomatched.append(ws[0])
        #     if debug:
        #         if len(ws)>1:
        #             print("A lot of windows!")
        #             for w in ws:
        #                 print("Windex: {}, seed et: {}, calo en: {}, calo eta: {}, seed eta: {}".format(w, 
        #                             windows_map[w]["et_seed"], calo_simenergy[calo], calo_simeta[calo], 
        #                             windows_map[w]["seed_eta"]))
        
        # In this version we keep all the windows

        ## Now save only the first N nocalomatched windows and then nocalowNMax of random ones
        metadata["n_windows_matched"] = len(windows_calomatched) 
        metadata["n_windows_nomatched"] = len(windows_nocalomatched)
        if self.nocalowNmax == 0:
            windows_to_keep_index = windows_calomatched
        elif len(windows_nocalomatched) > len(windows_calomatched):
            windows_to_keep_index = windows_calomatched + windows_nocalomatched[:len(windows_calomatched)] + \
                          random.sample(windows_nocalomatched[len(windows_calomatched):], min(self.nocalowNmax,len(windows_nocalomatched) - len(windows_calomatched) ))
        else:
            windows_to_keep_index = windows_calomatched + windows_nocalomatched

        if debug: print("Windows to keep: ", windows_to_keep_index)
        
        windows_to_keep = list(filter(lambda w: w["window_index"] in windows_to_keep_index, windows_map.values()))

       
        output_data = []

        for window in windows_to_keep:
            outw = {k:v for k,v in window.items() if k not in ["seed","seed_index"]}
            # outw["clusters"] = self.summary_clusters_window(window)
            # let's keep AOS approach 
            output_data.append(json.dumps(outw))
            # pprint(window)

        # if debug: print(output_data)
        return output_data, metadata

