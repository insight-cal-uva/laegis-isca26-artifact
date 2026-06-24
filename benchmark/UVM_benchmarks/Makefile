all:
	cd BN && make &
	cd CNN && make &
	cd knn && make &
	cd logistic-regression && make &
	cd SVM && make &
	cd rodinia/backprop && make &
	cd rodinia/dwt2d && make &
	cd rodinia/hotspot && make &
	cd rodinia/hotspot3D && make &
	cd rodinia/nw && make &
	cd rodinia/particlefilter && make &
	cd rodinia/pathfinder && make &
	cd rodinia/streamcluster && make &
	cd polybench; ./compileCodes.sh &
	wait

clean:
	cd BN && make clean && rm -f *.csv &
	cd CNN && make clean && rm -f *.csv &
	cd knn && make clean && rm -f *.csv &
	cd logistic-regression && make clean && rm -f *.csv &
	cd SVM && make clean && rm -f *.csv &
	cd rodinia/backprop && make clean && rm -f *.csv &
	cd rodinia/dwt2d && make clean && rm -f *.csv &
	cd rodinia/hotspot && make clean && rm -f *.csv &
	cd rodinia/hotspot3D && make clean && rm -f *.csv &
	cd rodinia/nw && make clean && rm -f *.csv &
	cd rodinia/particlefilter && make clean && rm -f *.csv &
	cd rodinia/pathfinder && make clean && rm -f *.csv &
	cd rodinia/streamcluster && make clean && rm -f *.csv &
	wait


get_permission:
	cd BN && chmod +x run &
	cd CNN && chmod +x run &
	cd knn && chmod +x run &
	cd logistic-regression && chmod +x run &
	cd SVM && chmod +x run &
	cd rodinia/backprop && chmod +x run &
	cd rodinia/dwt2d && chmod +x run &
	cd rodinia/hotspot && chmod +x run &
	cd rodinia/hotspot3D && chmod +x run &
	cd rodinia/nw && chmod +x run &
	cd rodinia/particlefilter && chmod +x run &
	cd rodinia/pathfinder && chmod +x run &
	cd rodinia/streamcluster && chmod +x run &
	cd polybench/2DCONV && chmod +x run &
	cd polybench/2MM && chmod +x run &
	cd polybench/3DCONV && chmod +x run &
	cd polybench/3MM && chmod +x run &
	cd polybench/ATAX && chmod +x run &
	cd polybench/BICG && chmod +x run &
	cd polybench/CORR && chmod +x run &
	cd polybench/COVAR && chmod +x run &
	cd polybench/FDTD-2D && chmod +x run &
	cd polybench/GEMM-100MB && chmod +x run &
	cd polybench/GESUMMV && chmod +x run &
	cd polybench/GRAMSCHM && chmod +x run &
	cd polybench/MVT && chmod +x run &
	cd polybench/SYR2K && chmod +x run &
	cd polybench/SYRK && chmod +x run &
	wait