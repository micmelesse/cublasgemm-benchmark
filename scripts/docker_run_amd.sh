alias drun='sudo docker run -it --rm --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $HOME/dockerx:/dockerx -w /dockerx/rocBLAS'
drun rocm/tensorflow:rocm3.0-tf1.15-python3 $1
