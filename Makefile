HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
        HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

EXE=dgemm_hip_rocm
CXXFLAGS = -O3 -g -I/opt/rocm/hipblas/include -lhipblas

$(EXE): dgemm_hip_rocm.cpp
        $(HIPCC) $(CXXFLAGS) $^ -o $@

clean:
        rm -rf *.o $(EXE)

