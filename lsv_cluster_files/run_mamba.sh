docker build -f ./lsv_cluster_files/mamba.dockerfile --build-arg USER_UID=$UID --build-arg USER_NAME=$(id -un) -t docker.lsv.uni-saarland.de/hhwang/mamba:0 .

