from __future__ import print_function
from math import pi, sqrt, cosh
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
    return data

class WindowCreator():

    def __init__(self, simfraction_thresholds,  seed_min_fraction=1e-2, cl_min_fraction=1e-4, simenergy_pu_limit = 1.5):
        self.seed_min_fraction = seed_min_fraction
        self.cluster_min_fraction = cl_min_fraction
        self.simfraction_thresholds = simfraction_thresholds
        self.simenergy_pu_limit = simenergy_pu_limit
        self.simfraction_thresholds = correctionlib.CorrectionSet.from_file(simfraction_thresholds)["simfraction_thres"]

    def pass_simfraction_threshold(self, seed_eta, seed_et, cluster_calo_score ):
        '''
        This functions associates a cluster as true matched if it passes a threshold in simfraction
        '''
        minscore = self.simfraction_thresholds.evaluate(seed_et, abs(seed_eta))
        return cluster_calo_score >= minscore

    def get_clusters_inside_window(self,seed_et, seed_eta, seed_phi, seed_iz, cls_eta, cls_phi, cls_iz,
                                   pfcluster_calo_map, pfcluster_calo_score, caloindex):
        true_cls = []
        cls_in_window = [ ]
        #######
        # Loop on all the clusters to find the ones in the windows and count the total and the true ones
        ######
        for icl in range(len(cls_eta)):
            cl_eta = cls_eta[icl]
            cl_phi = cls_phi[icl]
            cl_iz = cls_iz[icl] 
            isin, (etaw, phiw) = in_window(seed_eta,seed_phi,seed_iz, cl_eta, cl_phi, cl_iz,
                                           *self.dynamic_window(seed_eta))
            if isin:
                cls_in_window.append(icl)
                is_calo_matched =  pfcluster_calo_map[icl] == caloindex  # we know at this point it is not -1
                if is_calo_matched:
                    #associate the cluster to the caloparticle with simfraction optimized thresholds 
                    pass_simfrac_thres = self.pass_simfraction_threshold(seed_eta, 
                                                                         seed_et, pfcluster_calo_score[icl] )
                    if pass_simfrac_thres:
                        true_cls.append(icl)
        return cls_in_window, true_cls
                    
    def dynamic_window(self,eta):
        ## This is version 1
        aeta = abs(eta)

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



    def get_windows(self, event, assoc_strategy,  nocalowNmax, min_et_seed=1,
                    sc_collection="superCluster", reco_collection="none",
                    overwrite_runid = None,
                    loop_on_calo=False,  debug=False,):
        ## output
        available_branches = [f.GetName() for f in event.GetListOfBranches()]
        output_object = []
        output_event = []
        # Branches
        pfCluster_energy = event.pfCluster_energy
        pfCluster_rawEnergy = event.pfCluster_rawEnergy
        pfCluster_eta = event.pfCluster_eta
        pfCluster_phi = event.pfCluster_phi
        pfCluster_ieta = event.pfCluster_ieta
        pfCluster_iphi = event.pfCluster_iphi
        pfCluster_iz = event.pfCluster_iz

        calo_in_tree = "caloParticle_simEnergy" in available_branches
        if calo_in_tree:
            calo_simenergy = event.caloParticle_simEnergy
            calo_simenergy_goodstatus = event.caloParticle_simEnergyGoodStatus
            calo_genenergy = event.caloParticle_genEnergy
            calo_simeta = event.caloParticle_simEta
            calo_simphi = event.caloParticle_simPhi
            calo_geneta = event.caloParticle_genEta
            calo_genphi = event.caloParticle_genPhi
            calo_genpt = event.caloParticle_genPt
            calo_simiz = event.caloParticle_simIz
        # calo_geniz = event.caloParticle_genIzt
        # calo_isPU = event.caloParticle_isPU
        # calo_isOOTPU = event.caloParticle_isOOTPU
        pfcl_nxtals = event.pfCluster_nXtals
        nVtx = event.nVtx
        rho = event.rho
        obsPU = event.obsPU
        truePU = event.truePU
        #SuperCluster branches
        sc_rawEn = getattr(event, f"{sc_collection}_rawEnergy")
        #sc_rawESEn = event.superCluster_rawESEnergy
        sc_corrEn = getattr(event,f"{sc_collection}_energy")
        sc_eta = getattr(event,f"{sc_collection}_eta")
        sc_phi = getattr(event,f"{sc_collection}_phi")
        sc_nCls = getattr(event,f"{sc_collection}_nPFClusters")
        sc_seedIndex = [s for s in getattr(event,f"{sc_collection}_seedIndex")]
        pfcl_in_sc = getattr(event,f"{sc_collection}_pfClustersIndex")
        
        # GenParticle info
        genpart_energy = event.genParticle_energy
        genpart_eta = event.genParticle_eta
        genpart_phi = event.genParticle_phi
        genpart_pt = event.genParticle_pt
        genpart_pdgId = event.genParticle_pdgId
        genpart_status = event.genParticle_status
        genpart_statusFlag = event.genParticle_statusFlag if "genParticle_statusFlag" in available_branches else None
        genpart_good = []
        for igen in range(len(genpart_status)):
            if genpart_status[igen] != 1: continue
            if genpart_statusFlag != None:
                if genpart_statusFlag[igen] & 1 != 1:
                    continue
            if abs(genpart_pdgId[igen]) in [11,22]:
                genpart_good.append(igen)
        
                
        ## GenParticle SuperCluster matching
        # deltaR genParticle and SuperCluster seed
        genParticle_superCluster_matching = {}
        superCluster_genParticle_matching = {}        
        for igen in genpart_good:
            dR = []
            for isc in range(len(sc_eta)):
                dR.append((DeltaR(genpart_phi[igen],genpart_eta[igen],
                                  pfCluster_phi[sc_seedIndex[isc]],pfCluster_eta[sc_seedIndex[isc]] ),
                            isc))
            if dR:
                best_match = list(sorted(dR, key=itemgetter(0)))[0]
                if best_match[0] > 0.2:
                    # print(f"No SC matched to genPart {igen}")
                    continue
                genParticle_superCluster_matching[igen] = best_match[1]
                superCluster_genParticle_matching[best_match[1]] = igen
                
        #Gen association - patElectrons
        genParticle_patElectron_matching = {}
        patElectron_genParticle_matching = {}
        genParticle_patElectron_matching_second = {}
        patElectron_genParticle_matching_second = {} 
        for igen in genpart_good:
            dR = []
            for iele in range(len(event.patElectron_energy)):
                #print(f"PatElectron genphi {genpart_phi[igen]}, geneta{genpart_eta[igen]},elephi {event.patElectron_phi[iele]},eleeta={event.patElectron_eta[iele]}, ele{iele}")
                dR.append((DeltaR(genpart_phi[igen],genpart_eta[igen],
                                  event.patElectron_phi[iele],event.patElectron_eta[iele] ),
                            iele))
            if dR:
                sortedDR = list(sorted(dR, key=itemgetter(0)))
                best_match = sortedDR[0]
                #print(f"igen: {igen} eta: {genpart_eta[igen]:.2f} pt: {genpart_energy[igen]/np.cosh(genpart_eta[igen]):.2f}", sortedDR)    

                
                genParticle_patElectron_matching[igen] = (best_match[1], best_match[0])
                patElectron_genParticle_matching[best_match[1]] = (igen, best_match[0]) # saving the dR
                
                if len(sortedDR) > 1 and sortedDR[1][0] < 1.5:
                    genParticle_patElectron_matching_second[igen] = (sortedDR[1][1], sortedDR[1][0])
                    patElectron_genParticle_matching_second[sortedDR[1][1]] = (igen, sortedDR[1][0])
                    
                
                
        #Gen association - patPhotons
        genParticle_patPhoton_matching = {}
        patPhoton_genParticle_matching = {}
        genParticle_patPhoton_matching_second = {}
        patPhoton_genParticle_matching_second = {}
        for igen in genpart_good:
            dR = []
            for ipho in range(len(event.patPhoton_energy)):
                #print(f"patPhoton genphi {genpart_phi[igen]}, geneta{genpart_eta[igen]},elephi {event.patPhoton_phi[ipho]},eleeta={event.patPhoton_eta[ipho]}, pho{ipho}")
                dR.append((DeltaR(genpart_phi[igen],genpart_eta[igen],
                                  event.patPhoton_phi[ipho],event.patPhoton_eta[ipho] ),
                           ipho))
            if dR:
                sortedDR = list(sorted(dR, key=itemgetter(0)))
                best_match = sortedDR[0]
                #print(f"igen: {igen} pt: {genpart_energy[igen]/np.cosh(genpart_eta[igen])}", sortedDR)    
                            
                genParticle_patPhoton_matching[igen] = (best_match[1], best_match[0])
                patPhoton_genParticle_matching[best_match[1]] = (igen, best_match[0]) # saving the dR

                if len(sortedDR) > 1 and sortedDR[1][0] < 1.5:
                    genParticle_patPhoton_matching_second[igen] = (sortedDR[1][1], sortedDR[1][0])
                    patPhoton_genParticle_matching_second[sortedDR[1][1]] = (igen, sortedDR[1][0])

                
        # Map of seedRawId neeed to match electron/photon with SC
        superCluster_seedRawId_map = {}
        for sc, rawid in enumerate(event.superCluster_seedRawId):
            superCluster_seedRawId_map[rawid] = sc


        if calo_in_tree:
            clusters_scores = getattr(event, "pfCluster_"+assoc_strategy)
            # Get Association between pfcluster and calo
            # Sort the clusters for each calo in order of score. 
            # # This is needed to understand which cluster is the seed of the calo
            # Working only on signal caloparticle
            pfcluster_calo_map, pfcluster_calo_score, calo_pfcluster_map = \
                calo_association.get_calo_association(clusters_scores, sort_calo_cl=True,
                                                      debug=False, min_sim_fraction=self.cluster_min_fraction)

            calo_seeds_init = [(cls[0][0], calo)  for calo, cls in calo_pfcluster_map.items()] # [(seed,caloindex),..]
            calo_seeds =[]
            # clean the calo seeds
            for calo_seed, icalo in calo_seeds_init:
                # Check the minimum sim fraction
                if pfcluster_calo_score[calo_seed] < self.seed_min_fraction:
                    print("Seed score too small")
                    continue
                # Check if the calo and the seed are in the same window
                calo_inwindow = in_window(calo_geneta[icalo],calo_genphi[icalo],
                                          calo_simiz[icalo], pfCluster_eta[calo_seed],
                                          pfCluster_phi[calo_seed], pfCluster_iz[calo_seed],
                                          *self.dynamic_window(pfCluster_eta[calo_seed]))
                if not calo_inwindow:
                    print("Seed not in window")
                    continue 
                calo_seeds.append(calo_seed)

        #print("SuperCluster seeds index: ", sc_seedIndex)
        #print("Calo-seeds index: ", calo_seeds)
        calomatched_final_sc = [ ]
        ##################
        ## Object level info
        # Loop on the superCluster and check if they are genMatched or caloMatched
        if not loop_on_calo:

            # Analyze the superCluster if not reco_collection
            if reco_collection == "none":
                # Look on all the SuperClusters and check if the seed is a calo-seed
                for iSC in range(len(sc_rawEn)):
                    seed = sc_seedIndex[iSC]
                    calomatched = seed in calo_seeds
                    if calomatched: calomatched_final_sc.append(seed)
                    caloindex = pfcluster_calo_map[seed] if calomatched else -999
                    genmatched = iSC in  superCluster_genParticle_matching
                    genindex = superCluster_genParticle_matching[iSC] if genmatched else -999
                    #print(seed, calomatched, genmatched)

                    seed_eta = pfCluster_eta[seed]

                    seed_phi = pfCluster_phi[seed]
                    seed_iz = pfCluster_iz[seed]
                    seed_en = pfCluster_rawEnergy[seed]
                    seed_et = pfCluster_rawEnergy[seed] / cosh(pfCluster_eta[seed])

                    cls_in_window, true_cls = self.get_clusters_inside_window(seed_et, seed_eta, seed_phi, seed_iz,
                                                                              pfCluster_eta,  pfCluster_phi, pfCluster_iz,
                                                                              pfcluster_calo_map, pfcluster_calo_score, caloindex)
                    missing_cls, correct_cls, spurious_cls = [],[],[]
                    for icl in cls_in_window:
                        if icl in true_cls:
                            if icl in pfcl_in_sc[iSC]:
                                correct_cls.append(icl)
                            else:
                                missing_cls.append(icl)
                        else:
                            if icl in pfcl_in_sc[iSC]:
                                spurious_cls.append(icl)

                    out = {
                        "calomatched" : int(calomatched),
                        "caloindex": caloindex,
                        "genmatched" : int(genmatched),
                        "genindex": genindex ,
                        "sc_index": iSC,
                        "seed_index": seed,

                        "en_seed": pfCluster_rawEnergy[seed],
                        "et_seed": seed_et,
                        "en_seed_calib": pfCluster_energy[seed],
                        "et_seed_calib": pfCluster_energy[seed] / cosh(pfCluster_eta[seed]),
                        "seed_eta": seed_eta,
                        "seed_phi": seed_phi,
                        "seed_iz": seed_iz, 

                        "ncls_sel": sc_nCls[iSC],
                        "ncls_sel_true": len(correct_cls),
                        "ncls_sel_false": len(spurious_cls),
                        "ncls_true": len(true_cls),
                        "ncls_tot": len(cls_in_window),
                        "ncls_missing": len(missing_cls),

                        "en_sc_raw": sc_rawEn[iSC], 
                        "et_sc_raw": sc_rawEn[iSC]/cosh(sc_eta[iSC]),
                        #"en_sc_raw_ES" : sc_rawESEn[iSC],
                        #"et_sc_raw_ES" : sc_rawESEn[iSC]/ cosh(sc_eta[iSC]),
                        "en_sc_calib": sc_corrEn[iSC], 
                        "et_sc_calib": sc_corrEn[iSC]/cosh(sc_eta[iSC]), 

                        # Sim energy and Gen Enerugy of the caloparticle
                        "calo_en_gen": calo_genenergy[caloindex] if calomatched else -1, 
                        "calo_et_gen": calo_genenergy[caloindex]/cosh(calo_geneta[caloindex]) if calomatched else -1,
                        "calo_en_sim": calo_simenergy_goodstatus[caloindex] if calomatched else -1, 
                        "calo_et_sim": calo_simenergy_goodstatus[caloindex]/cosh(calo_geneta[caloindex]) if calomatched else -1,
                        "calo_geneta": calo_geneta[caloindex] if calomatched else -1,
                        "calo_genphi": calo_genphi[caloindex] if calomatched else -1,
                        "calo_simeta": calo_simeta[caloindex] if calomatched else -1,
                        "calo_simphi": calo_simphi[caloindex] if calomatched else -1,
                        "calo_genpt": calo_genpt[caloindex] if calomatched else -1,

                        # GenParticle info
                        "genpart_en": genpart_energy[genindex] if genmatched else -1,
                        "genpart_et": genpart_energy[genindex]/cosh(genpart_eta[genindex]) if genmatched else -1,
                        "gen_eta": genpart_eta[genindex] if genmatched else -1,
                        "gen_phi": genpart_phi[genindex] if genmatched else -1,
                        "gen_pt": genpart_pt[genindex] if genmatched else -1,


                        # PU information
                        "nVtx": nVtx, 
                        "rho": rho,
                        "obsPU": obsPU, 
                        "truePU": truePU,

                        #Evt number
                        "eventId" : event.eventId,
                        "runId": event.runId

                    }
                    output_object.append(out)

            ## Analyze the reco_collection
            elif reco_collection == "electron":
                for iEle in range(len(event.patElectron_index)):
                    seedRawId = event.patElectron_seedRawId[iEle]
                    cls_in_window, missing_cls, correct_cls, spurious_cls, true_cls = [],[],[],[],[]
                    calomatched = False
                    genmatched = iEle in patElectron_genParticle_matching
                    genindex = patElectron_genParticle_matching[iEle] if genmatched else -999
                    if seedRawId in superCluster_seedRawId_map:
                        sc_matched = 1
                        iSC = superCluster_seedRawId_map[seedRawId]
                        seed = sc_seedIndex[iSC]

                        if calo_in_tree:
                            calomatched = seed in calo_seeds
                            caloindex = pfcluster_calo_map[seed] if calomatched else -999
                        else:
                            calomatched = False
                            caloindex = -999

                        
                    
                        seed_eta = pfCluster_eta[seed]
                        seed_phi = pfCluster_phi[seed]
                        seed_iz = pfCluster_iz[seed]
                        seed_en = pfCluster_rawEnergy[seed]
                        seed_et = pfCluster_rawEnergy[seed] / cosh(pfCluster_eta[seed])

                        if calo_in_tree:
                            cls_in_window, true_cls = self.get_clusters_inside_window(seed_et, seed_eta, seed_phi, seed_iz,
                                                                              pfCluster_eta,  pfCluster_phi, pfCluster_iz,
                                                                              pfcluster_calo_map, pfcluster_calo_score, caloindex)
                            for icl in cls_in_window:
                                if icl in true_cls:
                                    if icl in pfcl_in_sc[iSC]:
                                        correct_cls.append(icl)
                                    else:
                                        missing_cls.append(icl)
                                else:
                                    if icl in pfcl_in_sc[iSC]:
                                        spurious_cls.append(icl)
                    else:
                        # There is no matched supercluster
                        sc_matched = 0
                        calomatched = False
                    
                        # print(f"Unmatched electron: {seedRawId}, eta: {event.patElectron_eta[iEle]}, et: { event.patElectron_et[iEle]}, trackerseeded: {event.patElectron_isEcalDriven[iEle]}")
                                    

                    out = {
                        "ele_index" : iEle,
                        "sc_matched" : sc_matched, 
                        "calomatched" : int(calomatched),
                        "caloindex": caloindex if calomatched else -999,
                        "sc_index": iSC if sc_matched else -199,
                        "seed_index": seed if sc_matched else -999,

                        "genmatched" : int(genmatched) if genmatched else 0,
                        "genindex": genindex if genmatched else -999,
                        
                        "en_seed": pfCluster_rawEnergy[seed] if sc_matched else -999,
                        "et_seed": seed_et if sc_matched else -999,
                        "en_seed_calib": pfCluster_energy[seed] if sc_matched else -999,
                        "et_seed_calib": pfCluster_energy[seed] / cosh(pfCluster_eta[seed]) if sc_matched else -999,
                        "seed_eta": seed_eta if sc_matched else -999,
                        "seed_phi": seed_phi if sc_matched else -999,
                        "seed_iz": seed_iz if sc_matched else -999, 

                        "ele_eta" : event.patElectron_eta[iEle],
                        "ele_phi" : event.patElectron_phi[iEle],
                        "ele_energy": event.patElectron_energy[iEle],
                        "ele_et": event.patElectron_et[iEle],
                        "ele_ecalEnergy": event.patElectron_ecalEnergy[iEle],
                        "ele_ecalSCEnergy": event.patElectron_ecalSCEnergy[iEle],
                        "ele_scRawEnergy": event.patElectron_ecalSCRawEnergy[iEle],
                        "ele_scRawESEnergy": event.patElectron_ecalSCRawESEnergy[iEle],
                        "ele_SCfbrem" : event.patElectron_superClusterFbrem[iEle],
                        "ele_tracfbrem" : event.patElectron_trackFbrem[iEle],
                        "ele_e5x5": event.patElectron_scE5x5[iEle],
                        "ele_e3x3": event.patElectron_scE3x3[iEle],
                        "ele_sigmaIEtaIEta": event.patElectron_scSigmaIEtaIEta[iEle],
                        "ele_sigmaIEtaIPhi" : event.patElectron_scSigmaIEtaIPhi[iEle],
                        "ele_sigmaIPhiIPhi" : event.patElectron_scSigmaIPhiIPhi[iEle],

                        "ele_ecalIso03": event.patElectron_ecalIso03[iEle],
                        "ele_trkIso03": event.patElectron_trkIso03[iEle],
                        "ele_hcalIso03": event.patElectron_hcalIso03[iEle],
                        "ele_pfChargedHadronIso" : event.patElectron_pfChargedHadronIso[iEle],
                        "ele_pfNeutralHadronIso" : event.patElectron_pfNeutralHadronIso[iEle],
                        "ele_pfPhotonIso" : event.patElectron_pfPhotonIso[iEle],
                        
                        "ele_isEcalDriven" : int(event.patElectron_isEcalDriven[iEle]) ,
                        "ele_isTrackerDriven" : int(event.patElectron_isTrackerDriven[iEle]),

                        "ele_HoE" : event.patElectron_HoE[iEle],
                        "ele_deltaEtaSeedClusterAtCalo": event.patElectron_deltaEtaSeedClusterAtCalo[iEle],
                        "ele_deltaPhiSeedClusterAtCalo": event.patElectron_deltaEtaSeedClusterAtCalo[iEle],
                        "ele_deltaEtaEleClusterAtCalo": event.patElectron_deltaEtaEleClusterAtCalo[iEle],
                        "ele_deltaPhiEleClusterAtCalo": event.patElectron_deltaPhiEleClusterAtCalo[iEle],

                        "ele_egmMVAElectronIDtight": event.patElectron_egmMVAElectronIDtight[iEle] ,
                        "ele_egmMVAElectronIDloose": event.patElectron_egmMVAElectronIDloose[iEle] ,
                        "ele_egmMVAElectronIDmedium": event.patElectron_egmMVAElectronIDmedium[iEle] ,


                        "ele_pAtCalo": event.patElectron_pAtCalo[iEle] ,
                        "ele_deltaEtaIn": event.patElectron_deltaEtaIn[iEle] ,
                        "ele_deltaPhiIn": event.patElectron_deltaPhiIn[iEle] ,
                        # "ele_trkPModeErr": event.patElectron_trkPModeErr[iEle],
                        # "ele_trkPMode": event.patElectron_trkPMode[iEle],
                        # "ele_trkEtaMode": event.patElectron_trkEtaMode[iEle],
                        # "ele_trkPhiMode": event.patElectron_trkPhiMode[iEle],
                        
                        "ncls_sel": sc_nCls[iSC] if sc_matched else -999,
                        "ncls_sel_true": len(correct_cls),
                        "ncls_sel_false": len(spurious_cls),
                        "ncls_true": len(true_cls),
                        "ncls_tot": len(cls_in_window),
                        "ncls_missing": len(missing_cls),

                        "en_sc_raw": sc_rawEn[iSC] if sc_matched else -999, 
                        "et_sc_raw": sc_rawEn[iSC]/cosh(sc_eta[iSC]) if sc_matched else -999,
                        #"en_sc_raw_ES" : sc_rawESEn[iSC],
                        #"et_sc_raw_ES" : sc_rawESEn[iSC]/ cosh(sc_eta[iSC]),
                        "en_sc_calib": sc_corrEn[iSC] if sc_matched else -999, 
                        "et_sc_calib": sc_corrEn[iSC]/cosh(sc_eta[iSC]) if sc_matched else -999,
                        "sc_etaWidth": event.superCluster_etaWidth[iSC] if sc_matched else -999,
                        "sc_phiWidth": event.superCluster_phiWidth[iSC] if sc_matched else -999,

                        # Sim energy and Gen Enerugy of the caloparticle
                        "calo_en_gen": calo_genenergy[caloindex] if calomatched else -1, 
                        "calo_et_gen": calo_genenergy[caloindex]/cosh(calo_geneta[caloindex]) if calomatched else -1,
                        "calo_en_sim": calo_simenergy_goodstatus[caloindex] if calomatched else -1, 
                        "calo_et_sim": calo_simenergy_goodstatus[caloindex]/cosh(calo_geneta[caloindex]) if calomatched else -1,
                        "calo_geneta": calo_geneta[caloindex] if calomatched else -1,
                        "calo_genphi": calo_genphi[caloindex] if calomatched else -1,
                        "calo_simeta": calo_simeta[caloindex] if calomatched else -1,
                        "calo_simphi": calo_simphi[caloindex] if calomatched else -1,
                        "calo_genpt": calo_genpt[caloindex] if calomatched else -1,

                        # GenParticle info
                        "genpart_en": genpart_energy[genindex] if genmatched else -1,
                        "genpart_et": genpart_energy[genindex]/cosh(genpart_eta[genindex]) if genmatched else -1,
                        "genpart_eta": genpart_eta[genindex] if genmatched else -1,
                        "genpart_phi": genpart_phi[genindex] if genmatched else -1,
                        "genpart_pt": genpart_pt[genindex] if genmatched else -1,

                        # PU information
                        "nVtx": nVtx, 
                        "rho": rho,
                        "obsPU": obsPU, 
                        "truePU": truePU,

                        #Evt number
                        "eventId" : event.eventId,
                        "runId": event.runId if not overwrite_runid else overwrite_runid

                    }
                    output_object.append(out)

            elif reco_collection == "photon":
                for iPho in event.photon_index:
                    seedRawId = event.photon_seedRawId[iPho]
                    cls_in_window, missing_cls, correct_cls, spurious_cls, true_cls = [],[],[],[],[]
                    calomatched = False
                    genmatched = False
                    if seedRawId in superCluster_seedRawId_map:
                        sc_matched = True
                        iSC = superCluster_seedRawId_map[event.photon_seedRawId[iPho]]
                        seed = sc_seedIndex[iSC]
                        calomatched = seed in calo_seeds
                        caloindex = pfcluster_calo_map[seed] if calomatched else -999
                        
                        genmatched = iSC in superCluster_genParticle_matching
                        genindex = superCluster_genParticle_matching[iSC] if genmatched else -999
                        
                        seed_eta = pfCluster_eta[seed]
                        seed_phi = pfCluster_phi[seed]
                        seed_iz = pfCluster_iz[seed]
                        seed_en = pfCluster_rawEnergy[seed]
                        seed_et = pfCluster_rawEnergy[seed] / cosh(pfCluster_eta[seed])
                        
                        cls_in_window, true_cls = self.get_clusters_inside_window(seed_et, seed_eta, seed_phi, seed_iz,
                                                                                  pfCluster_eta,  pfCluster_phi, pfCluster_iz,
                                                                                  pfcluster_calo_map, pfcluster_calo_score, caloindex)
                        for icl in cls_in_window:
                            if icl in true_cls:
                                if icl in pfcl_in_sc[iSC]:
                                    correct_cls.append(icl)
                                else:
                                    missing_cls.append(icl)
                            else:
                                if icl in pfcl_in_sc[iSC]:
                                    spurious_cls.append(icl)

                        out = {
                            "pho_index" : iPho,
                            "sc_matched" : sc_matched,
                            "calomatched" : int(calomatched) if sc_matched else -999,
                            "caloindex": caloindex if sc_matched else -999,
                            "genmatched" : int(genmatched) if sc_matched else -999,
                            "genindex": genindex if sc_matched else -999 ,
                            "sc_index": iSC if sc_matched else -999,
                            "seed_index": seed if sc_matched else -999,
                            
                            "en_seed": pfCluster_rawEnergy[seed] if sc_matched else -999,
                            "et_seed": seed_et if sc_matched else -999,
                            "en_seed_calib": pfCluster_energy[seed] if sc_matched else -999,
                            "et_seed_calib": pfCluster_energy[seed] / cosh(pfCluster_eta[seed]) if sc_matched else -999,
                            "seed_eta": seed_eta if sc_matched else -999,
                            "seed_phi": seed_phi if sc_matched else -999,
                            "seed_iz": seed_iz if sc_matched else -999, 

                            "pho_eta" : event.photon_eta[iPho],
                            "pho_phi" : event.photon_phi[iPho],
                            "pho_energy": event.photon_energy[iPho],
                            "pho_et" : event.photon_et[iPho],
                            "pho_scRawEnergy": event.photon_scRawEnergy[iPho],
                            "pho_e5x5": event.photon_e5x5[iPho],
                            "pho_e3x3": event.photon_e3x3[iPho],
                            "pho_sigmaIEtaIEta": event.photon_sigmaIEtaIEta[iPho],
                            "pho_sigmaIEtaIPhi" : event.photon_sigmaIEtaIPhi[iPho],
                            "pho_sigmaIPhiIPhi" : event.photon_sigmaIPhiIPhi[iPho],
                            "pho_hademCone": event.photon_hademCone[iPho],
                            
                            "ncls_sel": sc_nCls[iSC] if sc_matched else -999,
                            "ncls_sel_true": len(correct_cls),
                            "ncls_sel_false": len(spurious_cls),
                            "ncls_true": len(true_cls),
                            "ncls_tot": len(cls_in_window),
                            "ncls_missing": len(missing_cls),

                            "en_sc_raw": sc_rawEn[iSC] if sc_matched else -999, 
                            "et_sc_raw": sc_rawEn[iSC]/cosh(sc_eta[iSC]) if sc_matched else -999,
                            #"en_sc_raw_ES" : sc_rawESEn[iSC],
                            #"et_sc_raw_ES" : sc_rawESEn[iSC]/ cosh(sc_eta[iSC]),
                            "en_sc_calib": sc_corrEn[iSC] if sc_matched else -999, 
                            "et_sc_calib": sc_corrEn[iSC]/cosh(sc_eta[iSC]) if sc_matched else -999, 

                            # Sim energy and Gen Enerugy of the caloparticle
                            "calo_en_gen": calo_genenergy[caloindex] if calomatched else -1, 
                            "calo_et_gen": calo_genenergy[caloindex]/cosh(calo_geneta[caloindex]) if calomatched else -1,
                            "calo_en_sim": calo_simenergy_goodstatus[caloindex] if calomatched else -1, 
                            "calo_et_sim": calo_simenergy_goodstatus[caloindex]/cosh(calo_geneta[caloindex]) if calomatched else -1,
                            "calo_geneta": calo_geneta[caloindex] if calomatched else -1,
                            "calo_genphi": calo_genphi[caloindex] if calomatched else -1,
                            "calo_simeta": calo_simeta[caloindex] if calomatched else -1,
                            "calo_simphi": calo_simphi[caloindex] if calomatched else -1,
                            "calo_genpt": calo_genpt[caloindex] if calomatched else -1,
                            
                            # GenParticle info
                            "genpart_en": genpart_energy[genindex] if genmatched else -1,
                            "genpart_et": genpart_energy[genindex]/cosh(genpart_eta[genindex]) if genmatched else -1,
                            "gen_eta": genpart_eta[genindex] if genmatched else -1,
                            "gen_phi": genpart_phi[genindex] if genmatched else -1,
                            "gen_pt": genpart_pt[genindex] if genmatched else -1,
                            
                            # PU information
                            "nVtx": nVtx, 
                            "rho": rho,
                            "obsPU": obsPU, 
                            "truePU": truePU,
                            
                            #Evt number
                            "eventId" : event.eventId,
                            "runId": event.runId
                            
                        }
                        output_object.append(out)

            ##############################################################
            ##############################################################
            #### Analyze the genParticles to get info about the matching
            elif reco_collection == "genparticle":
                for igen in genpart_good:
                    cls_in_window, missing_cls, correct_cls, spurious_cls, true_cls = [],[],[],[],[]

                    # Take the matched SC and ele
                    iSC = genParticle_superCluster_matching.get(igen, -999)
                    iEle, deltaR_genPart_ele =  genParticle_patElectron_matching.get(igen, [-999,999.])
                    iPho, deltaR_genPart_pho = genParticle_patPhoton_matching.get(igen, [-999, 999.])
                    iEle_second, deltaR_genPart_ele_second =  genParticle_patElectron_matching_second.get(igen, [-999,999.])
                    iPho_second, deltaR_genPart_pho_second = genParticle_patPhoton_matching_second.get(igen, [-999, 999.])
                    #print(iSC, iEle, iPho)
                
                    ele_matched = iEle != -999
                    pho_matched = iPho != -999
                    sel_true_energy = 0.
                    true_energy = 0.
                    selected_energy = 0.
                    missing_energy = 0.
                    spurious_energy = 0.
                    
                    if iSC != -999:
                        sc_matched = 1
                        seed = sc_seedIndex[iSC]

                        if calo_in_tree:
                            calomatched = seed in calo_seeds
                            caloindex = pfcluster_calo_map[seed] if calomatched else -999
                        else:
                            calomatched = False
                            caloindex = -999

                        seed_eta = pfCluster_eta[seed]
                        seed_phi = pfCluster_phi[seed]
                        seed_iz = pfCluster_iz[seed]
                        seed_en = pfCluster_rawEnergy[seed]
                        seed_et = pfCluster_rawEnergy[seed] / cosh(pfCluster_eta[seed])
                        
                        if calo_in_tree:
                            cls_in_window, true_cls = self.get_clusters_inside_window(seed_et, seed_eta, seed_phi, seed_iz,
                                                                              pfCluster_eta,  pfCluster_phi, pfCluster_iz,
                                                                              pfcluster_calo_map, pfcluster_calo_score, caloindex)
                            for icl in cls_in_window:
                                if icl in true_cls:
                                    if icl in pfcl_in_sc[iSC]:
                                        correct_cls.append(icl)
                                    else:
                                        missing_cls.append(icl)
                                else:
                                    if icl in pfcl_in_sc[iSC]:
                                        spurious_cls.append(icl)
                            true_energy = sum([ pfCluster_energy[icl] for  icl in true_cls])
                            sel_true_energy = sum([ pfCluster_energy[icl] for  icl in correct_cls])
                            missing_energy = sum([ pfCluster_energy[icl] for  icl in missing_cls])
                            spurious_energy = sum([ pfCluster_energy[icl] for  icl in spurious_cls])
                            selected_energy = sum([pfCluster_energy[icl] for icl in pfcl_in_sc[iSC]])
                    else:
                        # There is no matched supercluster
                        sc_matched = 0
                        calomatched = False

                    out = {
                        "genindex": igen,
                        "elematched": int(ele_matched),
                        "phomatched": int(pho_matched),
                        "ele_index" : iEle,
                        "pho_index": iPho,
                        "ele_index_2nd": iEle_second,
                        "pho_index_2nd": iPho_second,
                        "deltaR_genPart_ele": deltaR_genPart_ele,
                        "deltaR_genPart_pho": deltaR_genPart_pho,
                        "deltaR_genPart_ele_second": deltaR_genPart_ele_second,
                        "deltaR_genPart_pho_second": deltaR_genPart_pho_second,
                        "calomatched" : int(calomatched),
                        "caloindex": caloindex if calomatched else -999,
                        "sc_matched" : int(sc_matched), 
                        "sc_index": iSC if sc_matched else -199,
                        "seed_index": seed if sc_matched else -999,
                        
                        "en_seed": pfCluster_rawEnergy[seed] if sc_matched else -999,
                        "et_seed": seed_et if sc_matched else -999,
                        "en_seed_calib": pfCluster_energy[seed] if sc_matched else -999,
                        "et_seed_calib": pfCluster_energy[seed] / cosh(pfCluster_eta[seed]) if sc_matched else -999,
                        "seed_eta": seed_eta if sc_matched else -999,
                        "seed_phi": seed_phi if sc_matched else -999,
                        "seed_iz": seed_iz if sc_matched else -999,
                        "sc_eta": sc_eta[iSC] if sc_matched else -999,
                        "sc_phi": sc_phi[iSC] if sc_matched else -999,


                        "sc_swissCross": event.superCluster_swissCross[iSC] if sc_matched else -999,
                        "sc_r9": event.superCluster_r9[iSC] if sc_matched else -999,
                        "sc_sigmaIetaIeta": event.superCluster_sigmaIetaIeta[iSC] if sc_matched else -999,
                        "sc_sigmaIetaIphi": event.superCluster_sigmaIetaIphi[iSC] if sc_matched else -999,
                        "sc_sigmaIphiIphi": event.superCluster_sigmaIphiIphi[iSC] if sc_matched else -999,
                        "sc_e5x5": event.superCluster_e5x5[iSC] if sc_matched else -999,

                        "sc_swissCross_f5x5": event.superCluster_full5x5_swissCross[iSC] if sc_matched else -999,
                        "sc_r9_f5x5": event.superCluster_full5x5_r9[iSC] if sc_matched else -999,
                        "sc_sigmaIetaIeta_f5x5": event.superCluster_full5x5_sigmaIetaIeta[iSC] if sc_matched else -999,
                        "sc_sigmaIetaIphi_f5x5": event.superCluster_full5x5_sigmaIetaIphi[iSC] if sc_matched else -999,
                        "sc_sigmaIphiIphi_f5x5": event.superCluster_full5x5_sigmaIphiIphi[iSC] if sc_matched else -999,

                        "sc_e5x5_f5x5": event.superCluster_full5x5_e5x5[iSC] if sc_matched else -999,

                        "ele_eta" : event.patElectron_eta[iEle] if ele_matched else -999,
                        "ele_phi" : event.patElectron_phi[iEle] if ele_matched else -999,
                        "ele_energy": event.patElectron_energy[iEle] if ele_matched else -999,
                        "ele_et": event.patElectron_et[iEle] if ele_matched else -999,
                        "ele_ecalEnergy": event.patElectron_ecalEnergy[iEle] if ele_matched else -999,
                        "ele_ecalSCEnergy": event.patElectron_ecalSCEnergy[iEle] if ele_matched else -999,
                        "ele_scRawEnergy": event.patElectron_ecalSCRawEnergy[iEle] if ele_matched else -999,
                        "ele_scRawESEnergy": event.patElectron_ecalSCRawESEnergy[iEle] if ele_matched else -999,


                        "ele_eta_2nd": event.patElectron_eta[iEle_second] if iEle_second != -999 else -999,
                        "ele_phi_2nd" : event.patElectron_phi[iEle_second] if iEle_second != -999 else -999,
                        "ele_energy_2nd": event.patElectron_energy[iEle_second] if iEle_second != -999 else -999,
                        "ele_et_2nd": event.patElectron_et[iEle_second] if iEle_second != -999 else -999,
                        "ele_ecalEnergy_2nd": event.patElectron_ecalEnergy[iEle_second] if iEle_second != -999 else -999,
                        "ele_ecalSCEnergy_2nd": event.patElectron_ecalSCEnergy[iEle_second] if iEle_second != -999 else -999,
                        "ele_scRawEnergy_2nd": event.patElectron_ecalSCRawEnergy[iEle_second] if iEle_second != -999 else -999,
                        "ele_scRawESEnergy_2nd": event.patElectron_ecalSCRawESEnergy[iEle_second] if iEle_second != -999 else -999,

                        
                        "ele_SCfbrem" : event.patElectron_superClusterFbrem[iEle] if ele_matched else -999,
                        "ele_tracfbrem" : event.patElectron_trackFbrem[iEle] if ele_matched else -999,
                        "ele_e5x5": event.patElectron_scE5x5[iEle] if ele_matched else -999,
                        "ele_e3x3": event.patElectron_scE3x3[iEle] if ele_matched else -999,
                        "ele_sigmaIEtaIEta": event.patElectron_scSigmaIEtaIEta[iEle] if ele_matched else -999,
                        "ele_sigmaIEtaIPhi" : event.patElectron_scSigmaIEtaIPhi[iEle] if ele_matched else -999,
                        "ele_sigmaIPhiIPhi" : event.patElectron_scSigmaIPhiIPhi[iEle] if ele_matched else -999,

                        "ele_ecalIso03": event.patElectron_ecalIso03[iEle] if ele_matched else -999,
                        "ele_trkIso03": event.patElectron_trkIso03[iEle] if ele_matched else -999,
                        "ele_hcalIso03": event.patElectron_hcalIso03[iEle] if ele_matched else -999,
                        "ele_pfChargedHadronIso" : event.patElectron_pfChargedHadronIso[iEle] if ele_matched else -999,
                        "ele_pfNeutralHadronIso" : event.patElectron_pfNeutralHadronIso[iEle] if ele_matched else -999,
                        "ele_pfPhotonIso" : event.patElectron_pfPhotonIso[iEle] if ele_matched else -999,\

                        "ele_HoE" : event.patElectron_HoE[iEle] if ele_matched else -999,
                        "ele_scEoP": event.patElectron_scEoP[iEle]  if ele_matched else -999,
                        "ele_ecalSCEoP": event.patElectron_ecalSCEoP[iEle]  if ele_matched else -999,
                        "ele_EoverP": event.patElectron_EoverP[iEle]  if ele_matched else -999,
                        
                        "ele_deltaEtaInTrack": event.patElectron_deltaEtaIn[iEle]  if ele_matched else -999,
                        "ele_deltaPhiInTrack": event.patElectron_deltaPhiIn[iEle]  if ele_matched else -999,

                        "ele_deltaEtaSeedClusterAtCalo": event.patElectron_deltaEtaSeedClusterAtCalo[iEle] if ele_matched else -999,
                        "ele_deltaPhiSeedClusterAtCalo": event.patElectron_deltaEtaSeedClusterAtCalo[iEle] if ele_matched else -999,
                        "ele_deltaEtaEleClusterAtCalo": event.patElectron_deltaEtaEleClusterAtCalo[iEle] if ele_matched else -999,
                        "ele_deltaPhiEleClusterAtCalo": event.patElectron_deltaPhiEleClusterAtCalo[iEle] if ele_matched else -999,

                        "ele_egmMVAElectronIDtight": event.patElectron_egmMVAElectronIDtight[iEle]  if ele_matched else -999,
                        "ele_egmMVAElectronIDloose": event.patElectron_egmMVAElectronIDloose[iEle]  if ele_matched else -999,
                        "ele_egmMVAElectronIDmedium": event.patElectron_egmMVAElectronIDmedium[iEle]  if ele_matched else -999,

                        
                        "ele_isEcalDriven" : int(event.patElectron_isEcalDriven[iEle]) if ele_matched else 0,
                        "ele_isTrackerDriven" : int(event.patElectron_isTrackerDriven[iEle]) if ele_matched else 0,

                        
                        # "ele_clsAdded_eta": [i for i in event.patElectron_clsAdded_eta[iEle]] if ele_matched else [],
                        # "ele_clsAdded_phi": [i for i in event.patElectron_clsAdded_phi[iEle]] if ele_matched else [],
                        # "ele_clsAdded_energy": [i for i in event.patElectron_clsAdded_energy[iEle]] if ele_matched else [],
                        # "ele_clsRemoved_eta": [i for i in event.patElectron_clsRemoved_eta[iEle]] if ele_matched else [],
                        # "ele_clsRemoved_phi": [i for i in event.patElectron_clsRemoved_phi[iEle]] if ele_matched else [],
                        # "ele_clsRemoved_energy": [i for i in event.patElectron_clsRemoved_energy[iEle]] if ele_matched else [],
                                             
                        "ncls_sel": sc_nCls[iSC] if sc_matched else -999,
                        "ncls_sel_true": len(correct_cls),
                        "ncls_sel_false": len(spurious_cls),
                        "ncls_true": len(true_cls),
                        "ncls_tot": len(cls_in_window),
                        "ncls_missing": len(missing_cls),

                        "ele_nclsRefinedSC": event.patElectron_scNPFClusters[iEle] if ele_matched else -999,
                        "ele_nclsEcalSC": event.patElectron_ecalSCNPFClusters[iEle] if ele_matched else -999,
                        "ele_passConversionVeto": int(event.patElectron_passConversionVeto[iEle]) if ele_matched else -999,
                        "ele_nOverlapPhotons": event.patElectron_nOverlapPhotons[iEle] if ele_matched else -999,
                        "ele_overlapPhotonIndices": [i for i in event.patElectron_overlapPhotonIndices[iEle]] if ele_matched else -999,
                        "ele_trackPAtCalo": event.patElectron_pAtCalo[iEle] if ele_matched else -999,

                        "ele_trackDeltaEtaIn": event.patElectron_deltaEtaIn[iEle]  if ele_matched else -999,
                        "ele_trackDeltaPhiIn": event.patElectron_deltaPhiIn[iEle]  if ele_matched else -999,
                        "ele_trackDeltaEtaSeedClusterAtCalo": event.patElectron_deltaEtaSeedClusterAtCalo[iEle]  if ele_matched else -999,
                        "ele_trackDeltaEtaEleClusterAtCalo": event.patElectron_deltaEtaEleClusterAtCalo[iEle]  if ele_matched else -999,
                        "ele_trackDeltaPhiEleClusterAtCalo": event.patElectron_deltaPhiEleClusterAtCalo[iEle]  if ele_matched else -999,
                        "ele_trackDeltaPhiEleClusterAtCalo": event.patElectron_deltaPhiEleClusterAtCalo[iEle]  if ele_matched else -999,

                        "ele_misHits": event.patElectron_misHits[iEle]  if ele_matched else -999,
                        "ele_nAmbiguousGsfTracks" : event.patElectron_nAmbiguousGsfTracks[iEle]  if ele_matched else -999,
                        "ele_trackFbrem": event.patElectron_trackFbrem[iEle]  if ele_matched else -999,
                        "ele_superClusterFbrem": event.patElectron_superClusterFbrem[iEle]  if ele_matched else -999,
                        "ele_dz": event.patElectron_dz[iEle]  if ele_matched else -999,
                        "ele_dxy": event.patElectron_dxy[iEle]  if ele_matched else -999,
                        "ele_dzError": event.patElectron_dzError[iEle]  if ele_matched else -999,
                        "ele_dxyError": event.patElectron_dxyError[iEle]  if ele_matched else -999,
                        "ele_pOut": event.patElectron_pOut[iEle]  if ele_matched else -999,
                        "ele_pIn": event.patElectron_pIn[iEle]  if ele_matched else -999,
                        "ele_pAtCalo": event.patElectron_pAtCalo[iEle]  if ele_matched else -999,

                        "ele_isEBEEGap": int(event.patElectron_isEBEEGap[iEle]) if ele_matched else -999,
                        "ele_isEBEtaGap": int(event.patElectron_isEBEtaGap[iEle]) if ele_matched else -999,
                        "ele_isEBPhiGap": int(event.patElectron_isEBPhiGap[iEle]) if ele_matched else -999,
                        "ele_isEEDeeGap": int(event.patElectron_isEEDeeGap[iEle]) if ele_matched else -999,
                        "ele_isEERingGap": int(event.patElectron_isEERingGap[iEle]) if ele_matched else -999,

                        "ele_dnn_signal_Isolated": event.patElectron_dnn_signal_Isolated[iEle] if ele_matched else -999,
                        "ele_dnn_signal_nonIsolated": event.patElectron_dnn_signal_nonIsolated[iEle] if ele_matched else -999,
                        "ele_dnn_bkg_nonIsolated": event.patElectron_dnn_bkg_nonIsolated[iEle] if ele_matched else -999,
                        "ele_dnn_bkg_Tau": event.patElectron_dnn_bkg_Tau[iEle] if ele_matched else -999,
                        "ele_dnn_bkg_Photon": event.patElectron_dnn_bkg_Photon[iEle] if ele_matched else -999,
                        
                        "pho_eta" : event.patPhoton_eta[iPho] if pho_matched else -999,
                        "pho_phi" : event.patPhoton_phi[iPho] if pho_matched else -999,
                        "pho_energy": event.patPhoton_energy[iPho] if pho_matched else -999,
                        "pho_et" : event.patPhoton_et[iPho] if pho_matched else -999,
                        "pho_scRawEnergy": event.patPhoton_scRawEnergy[iPho] if pho_matched else -999,

                        "pho_eta_2nd": event.patPhoton_eta[iPho_second] if iPho_second != -999 else -999,
                        "pho_phi_2nd" : event.patPhoton_phi[iPho_second] if iPho_second != -999 else -999,
                        "pho_energy_2nd": event.patPhoton_energy[iPho_second] if iPho_second != -999 else -999,
                        "pho_et_2nd": event.patPhoton_et[iPho_second] if iPho_second != -999 else -999,
                        "pho_scRawEnergy_2nd": event.patPhoton_scRawEnergy[iPho_second] if iPho_second != -999 else -999,
                        
                        "pho_e5x5": event.patPhoton_scE5x5[iPho] if pho_matched else -999,
                        "pho_e3x3": event.patPhoton_scE3x3[iPho] if pho_matched else -999,
                        "pho_sigmaIEtaIEta": event.patPhoton_scSigmaIEtaIEta[iPho] if pho_matched else -999,
                        "pho_sigmaIEtaIPhi" : event.patPhoton_scSigmaIEtaIPhi[iPho] if pho_matched else -999,
                        "pho_sigmaIPhiIPhi" : event.patPhoton_scSigmaIPhiIPhi[iPho] if pho_matched else -999,
                        "pho_HoE": event.patPhoton_HoE[iPho] if pho_matched else -999,
                        
                        "pho_etOutsideMustache": event.patPhoton_etOutsideMustache[iPho] if pho_matched else -999,
                        "pho_nClusterOutsideMustache": event.patPhoton_nClusterOutsideMustache[iPho] if pho_matched else -999,
                        "pho_pfChargedHadronIso": event.patPhoton_pfChargedHadronIso[iPho] if pho_matched else -999,
                        "pho_pfNeutralHadronIso": event.patPhoton_pfNeutralHadronIso[iPho] if pho_matched else -999,
                        "pho_pfPhotonIso": event.patPhoton_pfPhotonIso[iPho] if pho_matched else -999,
                        "pho_patParticleIso": event.patPhoton_patParticleIso[iPho] if pho_matched else -999,
                        "pho_ecalIso03": event.patPhoton_ecalIso03[iPho] if pho_matched else -999,
                        "pho_hcalIso03": event.patPhoton_hcalIso03[iPho] if pho_matched else -999,
                        "pho_egmMVAPhotonIDmedium": event.patPhoton_egmMVAPhotonIDmedium[iPho] if pho_matched else -999,
                        "pho_egmMVAPhotonIDtight": event.patPhoton_egmMVAPhotonIDtight[iPho] if pho_matched else -999,

                        "pho_scNPFClusters": event.patPhoton_scNPFClusters[iPho] if pho_matched else -999,
                        "pho_ecalSCNPFClusters": event.patPhoton_ecalSCNPFClusters[iPho] if pho_matched else -999,

                        "pho_passElectronVeto": int(event.patPhoton_passElectronVeto[iPho]) if pho_matched else -999,
                        "pho_hasPixelSeed": int(event.patPhoton_hasPixelSeed[iPho]) if pho_matched else -999,
                        "pho_hasConversionTracks": int(event.patPhoton_hasConversionTracks[iPho]) if pho_matched else -999,

                        "cl2_en": pfCluster_energy[pfcl_in_sc[iSC][1]] if sc_matched and sc_nCls[iSC] >= 2 else -999,
                        "cl3_en": pfCluster_energy[pfcl_in_sc[iSC][2]] if sc_matched and sc_nCls[iSC] >= 3 else -999,
                        "cl4_en": pfCluster_energy[pfcl_in_sc[iSC][3]] if sc_matched and sc_nCls[iSC] >= 4 else -999,

                        "en_sc_raw": sc_rawEn[iSC] if sc_matched else -999, 
                        "et_sc_raw": sc_rawEn[iSC]/cosh(sc_eta[iSC]) if sc_matched else -999,
                        #"en_sc_raw_ES" : sc_rawESEn[iSC],
                        #"et_sc_raw_ES" : sc_rawESEn[iSC]/ cosh(sc_eta[iSC]),
                        "en_sc_calib": sc_corrEn[iSC] if sc_matched else -999, 
                        "et_sc_calib": sc_corrEn[iSC]/cosh(sc_eta[iSC]) if sc_matched else -999,
                        "sc_etaWidth": event.superCluster_etaWidth[iSC] if sc_matched else -999,
                        "sc_phiWidth": event.superCluster_phiWidth[iSC] if sc_matched else -999,

                        "true_energy_cls": true_energy,
                        "sel_true_energy_cls": sel_true_energy,
                        "missing_energy_cls": missing_energy,
                        "sel_false_energy_cls": spurious_energy,
                        "selected_energy_cls": selected_energy,

                        # Sim energy and Gen Enerugy of the caloparticle
                        "calo_en_gen": calo_genenergy[caloindex] if calomatched else -1, 
                        "calo_et_gen": calo_genenergy[caloindex]/cosh(calo_geneta[caloindex]) if calomatched else -1,
                        "calo_en_sim": calo_simenergy_goodstatus[caloindex] if calomatched else -1, 
                        "calo_et_sim": calo_simenergy_goodstatus[caloindex]/cosh(calo_geneta[caloindex]) if calomatched else -1,
                        "calo_geneta": calo_geneta[caloindex] if calomatched else -1,
                        "calo_genphi": calo_genphi[caloindex] if calomatched else -1,
                        "calo_simeta": calo_simeta[caloindex] if calomatched else -1,
                        "calo_simphi": calo_simphi[caloindex] if calomatched else -1,
                        "calo_genpt": calo_genpt[caloindex] if calomatched else -1,

                        # GenParticle info
                        "genpart_en": genpart_energy[igen],
                        "genpart_et": genpart_energy[igen]/cosh(genpart_eta[igen]),
                        "genpart_eta": genpart_eta[igen],
                        "genpart_phi": genpart_phi[igen],
                        "genpart_pt": genpart_pt[igen],

                        # PU information
                        "nVtx": nVtx, 
                        "rho": rho,
                        "obsPU": obsPU, 
                        "truePU": truePU,

                        #Evt number
                        "eventId" : event.eventId,
                        "runId": event.runId if not overwrite_runid else overwrite_runid

                    }
                    output_object.append(out)


        ##############################################  
        ##########################
        ## IF we want to loop on calo-matched seeds instead of SC object
        else:
            for seed in calo_seeds:
                calomatched = True
                # Check if there is a SuperCluster with this calo
                # (The calo seeds have been already filtered by simfraction and inWindow)
                sc_found = seed in sc_seedIndex
                if not sc_found : continue
                iSC = sc_seedIndex.index(seed)
                
                caloindex = pfcluster_calo_map[seed]
                genmatched = iSC in superCluster_genParticle_matching
                genindex = superCluster_genParticle_matching[iSC] if genmatched else -999
                
                seed_eta = pfCluster_eta[seed]
                seed_phi = pfCluster_phi[seed]
                seed_iz = pfCluster_iz[seed]
                seed_en = pfCluster_rawEnergy[seed]
                seed_et = pfCluster_rawEnergy[seed] / cosh(pfCluster_eta[seed])
                
                cls_in_window, true_cls = self.get_clusters_inside_window(seed_et, seed_eta, seed_phi, seed_iz,
                                                                     pfCluster_eta,  pfCluster_phi, pfCluster_iz,
                                                                          pfcluster_calo_map, pfcluster_calo_score, caloindex)
                missing_cls, correct_cls, spurious_cls = [],[],[]
                for icl in cls_in_window:
                    if icl in true_cls:
                        if icl in pfcl_in_sc[iSC]:
                            correct_cls.append(icl)
                        else:
                            missing_cls.append(icl)
                    else:
                        if icl in pfcl_in_sc[iSC]:
                            spurious_cls.append(icl)
                # print(true_cls, correct_cls, spurious_cls, missing_cls)
                            
                out = {
                    "calomatched" : 1,
                    "caloindex": caloindex,
                    "genmatched" : int(genmatched),
                    "genindex": genindex,
                    "sc_index": iSC,
                    "seed_index": seed,
                    
                    "en_seed": pfCluster_rawEnergy[seed],
                    "et_seed": seed_et,
                    "en_seed_calib": pfCluster_energy[seed],
                    "et_seed_calib": pfCluster_energy[seed] / cosh(pfCluster_eta[seed]),
                    "seed_eta": seed_eta,
                    "seed_phi": seed_phi,
                    "seed_iz": seed_iz, 
                                        
                    "ncls_sel": sc_nCls[iSC],
                    "ncls_sel_true": len(correct_cls),
                    "ncls_sel_false": len(spurious_cls),
                    "ncls_true": len(true_cls),
                    "ncls_tot": len(cls_in_window),
                    "ncls_missing": len(missing_cls),
                    
                    "en_sc_raw": sc_rawEn[iSC], 
                    "et_sc_raw": sc_rawEn[iSC]/cosh(sc_eta[iSC]),
                    #"en_sc_raw_ES" : sc_rawESEn[iSC],
                    #"et_sc_raw_ES" : sc_rawESEn[iSC]/ cosh(sc_eta[iSC]),
                    "en_sc_calib": sc_corrEn[iSC], 
                    "et_sc_calib": sc_corrEn[iSC]/cosh(sc_eta[iSC]), 
                    
                    # Sim energy and Gen Enerugy of the caloparticle
                    "calo_en_gen": calo_genenergy[caloindex] if calomatched else -1, 
                    "calo_et_gen": calo_genenergy[caloindex]/cosh(calo_geneta[caloindex]) if calomatched else -1,
                    "calo_en_sim": calo_simenergy_goodstatus[caloindex] if calomatched else -1, 
                    "calo_et_sim": calo_simenergy_goodstatus[caloindex]/cosh(calo_geneta[caloindex]) if calomatched else -1,
                    "calo_geneta": calo_geneta[caloindex] if calomatched else -1,
                    "calo_genphi": calo_genphi[caloindex] if calomatched else -1,
                    "calo_simeta": calo_simeta[caloindex] if calomatched else -1,
                    "calo_simphi": calo_simphi[caloindex] if calomatched else -1,
                    "calo_genpt": calo_genpt[caloindex] if calomatched else -1,
                    
                    # GenParticle info
                    "genpart_en": genpart_energy[genindex] if genmatched else -1,
                    "genpart_et": genpart_energy[genindex]/cosh(genpart_eta[genindex]) if genmatched else -1,
                    "gen_eta": genpart_eta[genindex] if genmatched else -1,
                    "gen_phi": genpart_phi[genindex] if genmatched else -1,
                    "gen_pt": genpart_pt[genindex] if genmatched else -1,
                    
                    
                    # PU information
                    "nVtx": nVtx, 
                    "rho": rho,
                    "obsPU": obsPU, 
                    "truePU": truePU,
                    
                    #Evt number
                    "eventId" : event.eventId,
                    "runId": event.runId
                    
                }
                output_object.append(out)
                    
        return output_object, output_event
