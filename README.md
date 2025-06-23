This repo contains the code for the gelato pipeline.

There are two directories: `trigger` and `offline`. `trigger` contains infrastruction for calculating AD scores and gelato trigger decisions. `offline` contains infrastructure for looking at distributions of offline objects passing the gelato triggers.

In either case, the first step is to use xAODAnaHelpers to read in AOD files (for trigger objects) or DAOD files (for offline objects) and collect the relevant variables into a smaller TTree. For this, one should use the following branch of xAODAnaHelpers: https://github.com/max-cohen54/xAODAnaHelpers/tree/merged_cohen_xAODAnaHelpers

xAH config and run files used are located in the trigger and offline dirs for their respective uses.

In order to emulate L1AD scores, we need a special version of QKeras. To set this up, first make sure you're part of the atlas-trig-anomdet email list (which will give you access to specific directories.) Next on lxplus, run `conda activate /eos/user/m/mmcohen/conda_envs/l1AD_env` to activate the environment.
