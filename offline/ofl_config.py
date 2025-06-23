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
                                    "m_triggerSelection": ".*VAE.*", # only save events that pass VAE chains
                                    "m_applyTriggerCut": True, # apply the trigger cut
                                    "m_storeTrigDecisions": True,
                                    "m_storePassL1": True,
                                    "m_storePassHLT": True,
                                    "m_name": "myBaseEventSel"})






# Saving Offline objects
c.algorithm("TreeAlgo", {
                         "m_debug": True,
                         "m_name": "output_tree", # name of the output TTree
                         "m_jetContainerName": "AntiKt4EMPFlowJets",
                         "m_jetBranchName": "jet",
                         "m_jetDetailStr": "kinematic clean timing energy",
                         "m_photonContainerName": "Photons",
                         "m_photonBranchName": "photon",
                         "m_photonDetailStr": "kinematic PID PID_Medium",
                         "m_elContainerName": "Electrons",
                         "m_elBranchName": "electron",
                         "m_elDetailStr": "kinematic PID PID_Medium",
                         "m_muContainerName": "Muons",
                         "m_muBranchName": "muon",
                         "m_muDetailStr": "kinematic quality RECO_Medium",
			             "m_evtDetailStr": "pileup",
                         "m_trigDetailStr": "basic passTriggers", # save trigger decisions
                         })


