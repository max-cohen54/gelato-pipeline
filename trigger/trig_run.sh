

submit="local" # local, grid
loglevel=info
parentsubmitdir="/eos/home-m/mmcohen/ad_trigger_development/ops/data/trees/" # better to use some location on eos that has a lot of space available
submitdir="06-11-2025_ops_local_498335_lb201"
nevents=-1 # -1: all
configFile="../configs/AD_chain_config.py"
#dataset="data25_13p6TeV.00496825.physics_Main.merge.AOD.f1575_m2272"
#outname="user.mmcohen.data25_496825_ops_05-07-2025"

if [ "$submit" == "grid" ]; then
    # grid
    lsetup panda
    voms-proxy-init -voms atlas
    #infile="/eos/home-m/mmcohen/temp/xaodanahelpersminimalexample/data/DAODs/data22_13p6TeV.00437548.physics_Main.deriv.DAOD_JETM1.f1302_m2142_p5415/DAOD_JETM1.31025456._000028.pool.root.1"
    cmd="xAH_run.py --files $dataset --config $configFile --inputRucio --submitDir ${parentsubmitdir}/${submitdir}_$outname $extra -f prun --optGridOutputSampleName $outname --optTmpDir /tmp --optSubmitFlags=\"--addNthFieldOfInDSToLFN=2\""
    $cmd
    echo $cmd

else
    # local
    infile="/eos/atlas/atlastier0/rucio/data25_13p6TeV/physics_Main/00498335/data25_13p6TeV.00498335.physics_Main.merge.AOD.f1586_m2272/data25_13p6TeV.00498335.physics_Main.merge.AOD.f1586_m2272._lb0201._0002.1"
    #infile="/eos/atlas/atlastier0/rucio/data25_13p6TeV/physics_Main/00497637/data25_13p6TeV.00497637.physics_Main.merge.AOD.x907_m2272/data25_13p6TeV.00497637.physics_Main.merge.AOD.x907_m2272._lb0186._0005.1"
    #infile="/eos/atlas/atlastier0/rucio/data25_13p6TeV/physics_Main/00497727/data25_13p6TeV.00497727.physics_Main.merge.AOD.x907_m2272/data25_13p6TeV.00497727.physics_Main.merge.AOD.x907_m2272._lb0299._0002.1"
    #infile="/eos/home-m/mmcohen/ad_trigger_development/ops/data/AODs/data25_13p6TeV.00497727.physics_Main.merge.AOD.x907_m2272/data25_13p6TeV.00497727.physics_Main.merge.AOD.x907_m2272._lb0114._0001.1"

    submitdir=${submitdir}_${nevents}
    mkdir -p $parentsubmitdir
    cmd="xAH_run.py --mode athena --files $infile --nevents $nevents --config $configFile --submitDir ${parentsubmitdir}/$submitdir --log-level $loglevel $extra --force direct"
    $cmd
    echo $cmd
fi

