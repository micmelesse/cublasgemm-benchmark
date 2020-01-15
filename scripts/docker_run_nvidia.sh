alias nv_drun='sudo docker run -it --network=host --runtime=nvidia --ipc=host -v $HOME/dockerx:/dockerx -w /dockerx/cublasgemm-benchmark'
nv_drun nvidia/cuda:10.1-devel-ubuntu16.04
