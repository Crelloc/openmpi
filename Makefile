CFLAGS:= -Wall -Werror -O0 -g -fprofile-arcs -ftest-coverage


all: output

mmm_mpi: mmm_mpi.o
	mpicc mmm_mpi.o -o mmm_mpi -lgcov
	
mmm_mpi.o: mmm_mpi.c
	mpicc $(CFLAGS) -c mmm_mpi.c
	
output: mmm_mpi test_2p_500n
	@echo
	@echo '*'
	@echo '* Generating HTML output'
	@echo '*'
	@echo
	mkdir ./output
	genhtml trace_args.info \
		   --output-directory ./output --title "code coverage" \
		   --show-details \
		   --legend
	@echo
	@echo '*'
	@echo "* See index.html"
	@echo '*'
	@echo
	
test_2p_500n:
	@echo
	@echo '*'
	@echo '* Test: running mpirun -np 2 ./mmm_mpi 500'
	@echo '*'
	@echo
	lcov --zerocounters --directory .
	mpirun -np 2 ./mmm_mpi 500
	lcov --capture --directory . --output-file trace_args.info --test-name test_2p_500n --no-external

clean:
	rm mmm_mpi *.o *.gcno *.gcda *.info
	rm -r ./output
