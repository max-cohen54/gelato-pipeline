In order to run the emulated gelato chain, one should do the following:

1) run the xAODAnaHelpers over the run that you want to look at. Use the trig_config.py and trig_run.sh files as an example. This runs over all events in the run, collecting trigger objects and other relevant variables into a smaller TTree.

2) use the make_ntuples notebook to turn the TTree into an h5 file

3) run the l1AD_inference sh script over the new h5 file. This script will add L1AD scores to the h5 file.

4) run the HLTAD_inference_and_plot notebook which will update the h5 file again and plot the results.