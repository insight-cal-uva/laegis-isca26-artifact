all:
	/usr/local/cuda/bin/nvcc -O3 ${CUFILES} -o ${EXECUTABLE} 
clean:
	rm -f *~ *.exe