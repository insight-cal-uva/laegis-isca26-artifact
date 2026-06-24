all:
	cd UVM_benchmarks && make &
	cd non_UVM_benchmarks && make &
	cd UVM_benchmarks_oversub && make &
	wait
	@echo "done"

clean:
	cd UVM_benchmarks && make clean &
	cd non_UVM_benchmarks && make clean &
	cd UVM_benchmarks_oversub && make clean &
	wait
	@echo "done"
