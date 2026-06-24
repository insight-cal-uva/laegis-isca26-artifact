include ../../common/make.config


CC := nvcc

all:
	$(CC) -arch=sm_90 -lcuda -lcublas *.cu -o CNN

run:
	./CNN
clean:
	rm -f CNN
