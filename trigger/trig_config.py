#adapted from: https://xaodanahelpers.readthedocs.io/en/latest/UsingUs.html

from xAODAnaHelpers import Config
c = Config()

c.algorithm("BasicEventSelection", {"m_truthLevelOnly": False,
                                    #"m_applyGRLCut": True,
                                    #"m_GRLxml": "/cvmfs/atlas.cern.ch/repo/sw/database/GroupData/GoodRunsLists/data24_13p6TeV/20241020/data24_13p6TeV.periodsEtoN_DetStatus-v128-pro36-07_MERGED_PHYS_StandardGRL_All_Good_25ns.xml",
                                    "m_doPUreweighting": False,
                                    "m_vertexContainerName": "PrimaryVertices",
                                    "m_PVNTrack": 2,
                                    "m_useMetaData": False,
                                    "m_triggerSelection": "L1_.*|HLT_.*",
                                    "m_applyTriggerCut": True,
                                    "m_storeTrigDecisions": True,
                                    "m_storePassL1": True,
                                    "m_storePassHLT": True,
                                    "m_name": "myBaseEventSel"})



c.algorithm("TreeAlgo", {
                         "m_debug": True,
                         "m_name": "output_tree",
                         "m_jetContainerName": "HLT_AntiKt4EMPFlowJets_subresjesgscIS_ftf",
                         "m_jetBranchName": "HLT_jet",
                         "m_jetDetailStr": "kinematic clean timing energy",
                         "m_photonContainerName": "HLT_egamma_Photons",
                         "m_photonBranchName": "HLT_ph",
                         "m_photonDetailStr": "kinematic PID PID_Medium",
                         "m_elContainerName": "HLT_egamma_Electrons_GSF HLT_egamma_Electrons_LRT",
                         "m_elBranchName": "HLT_el HLT_el_LRT",
                         "m_elDetailStr": "kinematic PID PID_Medium",
                         "m_muContainerName": "HLT_Muons_FS HLT_MuonsCB_LRT",
                         "m_muBranchName": "HLT_muon HLT_muon_LRT",
                         "m_muDetailStr": "kinematic quality RECO_Medium",
			             "m_l1JetContainerName": "L1_jFexSRJetRoI L1_jFexLRJetRoI",
                         "m_l1JetBranchName": "L1_jFexSRJet L1_jFexLRJet",
			             "m_TrigMETContainerName": "HLT_MET_tcpufit",
                         "m_TrigMETDetailStr": "kinematic",
                         "m_trigDetailStr": "basic passTriggers",
                         "m_l1MuonContainerName": "LVL1MuonRoIs",
                         "m_l1MuonBranchName": "L1Muon",
                         "m_l1TauContainerName": "L1_jFexTauRoI L1_eTauRoI",
                         "m_l1TauBranchName": "L1Tau_jFex L1Tau_eFex",
                         "m_l1EMContainerName": "L1_eEMRoI",
                         "m_l1EMBranchName": "L1Egamma",
                         "m_l1MetContainerName": "L1_jFexMETRoI",
                         "m_l1MetDetailStr": "kinematic",
                         "m_l1MetBranchName": "L1MET",
			             "m_evtDetailStr": "pileup"
                         })


