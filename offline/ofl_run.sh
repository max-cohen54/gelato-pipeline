

submit="grid" # local, grid
loglevel=info
parentsubmitdir="/eos/home-m/mmcohen/ad_trigger_development/ops/data/trees/" # good to use some location on eos that has a lot of space available
submitdir="06-22-2025_ad_chain_grid_498007"
nevents=-1 # -1: all
configFile="../configs/AD_chain_config.py" # config file to be used for the run
dataset="data25_13p6TeV.00498007.physics_Main.deriv.DAOD_PHYS.f1584_m2272_p6828" # if grid run, need to specify the sample name
outname="user.mmcohen.data25_498007_ad_chain_06-22-2025" # if grid run, need to specify the output name

if [ "$submit" == "grid" ]; then
    # grid
    lsetup panda
    voms-proxy-init -voms atlas
    cmd="xAH_run.py --files $dataset --config $configFile --inputRucio --submitDir ${parentsubmitdir}/${submitdir}_$outname $extra -f prun --optGridOutputSampleName $outname --optTmpDir /tmp --optSubmitFlags=\"--addNthFieldOfInDSToLFN=2\""
    $cmd
    echo $cmd

else
    # local

    # if local, need to give the local dir of the input file
    infile="/eos/home-m/mmcohen/temp/xaodanahelpersminimalexample/data/AODs/data25_13p6TeV.00498007.physics_Main.deriv.DAOD_PHYS.f1584_m2272_p6828/DAOD_PHYS.44853004._000252.pool.root.1"

    submitdir=${submitdir}_${nevents}
    mkdir -p $parentsubmitdir
    cmd="xAH_run.py --mode athena --files $infile --nevents $nevents --config $configFile --submitDir ${parentsubmitdir}/$submitdir --log-level $loglevel $extra --force direct"
    $cmd
    echo $cmd
fi

