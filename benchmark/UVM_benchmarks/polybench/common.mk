all:
	/usr/local/cuda/bin/nvcc -O3 ${CUFILES} ${DEF} -o ${EXECUTABLE} 
clean:
	rm -f *~ *.exe