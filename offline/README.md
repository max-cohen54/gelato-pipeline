to make distributions of the offline objects passing the gelato triggers, one should do the following:

1) run xAODAnaHelpers over the run that you want to look at. Use the ofl_config.py and ofl_run.sh files as an example. Currently, these only run over events passing any chain with 'VAE' in the name, which is only the GELATO chains, and saves offline objects.

2) use the check_offline_uniqeu_events.ipynb notebook as an example of how to read in the resulting TTree and make some plots 