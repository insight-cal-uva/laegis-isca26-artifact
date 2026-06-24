all:nv_drv_test

nv_drv_test: 
	nvcc drv_test_main.c -o nv_drv_test -lcudart
	nvcc simple.cu -o gpu_reg

clean:
	rm nv_drv_test