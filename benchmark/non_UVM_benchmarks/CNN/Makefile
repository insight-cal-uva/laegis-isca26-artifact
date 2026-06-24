all:
	nvcc -arch=sm_90  -lcuda -lcublas *.cu -o CNN   -Wno-deprecated-gpu-targets

run:
	./CNN
clean:
	rm -f CNN
