HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc 

EXE=dgemm_rocm_f64 dgemm_rocm_f32 dgemm_rocm_f16 dgemm_rocm_bf16
CXXFLAGS = -O3 -g -I/opt/rocm/hipblas/include -I/opt/rocm/include -lhipblas -lrocblas

all: $(EXE)

% : %.cpp
	$(HIPCC) $(CXXFLAGS) $^ -o $@

clean:
	rm -rf *.o $(EXE)
