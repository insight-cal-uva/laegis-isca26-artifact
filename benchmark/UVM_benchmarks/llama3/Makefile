NVCC = nvcc

all: runcuda

.PHONY: runcuda
runcuda:
# $(NVCC) -arch=sm_90 -g -o runcuda llama3.cu -lm -lcudart
# $(NVCC)  -arch=sm_90 -DUSE_CUBLAS=1 -g -o runcuda llama3.cu -lm -lcublas
	$(NVCC) -DUSE_CUBLAS=1 -g -o runcuda llama3.cu -lm -lcublas

.PHONY: clean
clean:
	rm -f runcuda _cuobj* _app_cuda* *.ptx*
