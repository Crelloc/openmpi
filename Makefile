CFLAGS:= -Wall -Werror -g
COVFLAGS:= -fprofile-arcs -ftest-coverage

all: mmm_mpi

mmm_mpi: mmm_mpi.o
	mpicc mmm_mpi.o -o mmm_mpi

mmm_mpi.o: mmm_mpi.c
	mpicc $(CFLAGS) -O2 -c mmm_mpi.c

TS:
	mpicc -D_TS_ $(CFLAGS) -O2 -o mmm_mpi -lgcov mmm_mpi.c

mmm_mpi_c:
	mpicc $(CFLAGS) $(COVFLAGS) -o mmm_mpi -lgcov mmm_mpi.c

mmm_mpi_d:
	mpicc $(CFLAGS) $(COVFLAGS) -O0 -o mmm_mpi -lgcov mmm_mpi.c

output: mmm_mpi_c test_2p_500n test_4p_500n
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

test_4p_500n:
	@echo
	@echo '*'
	@echo '* Test: running mpirun -n 4 ./mmm_mpi 500'
	@echo '*'
	@echo
	lcov --zerocounters --directory .
	mpirun -n 4 ./mmm_mpi 500
	lcov --capture --directory . --output-file trace_args.info --test-name test_4p_500n --no-external
debug_4p: mmm_mpi_d
	@echo
	@echo '*'
	@echo '* debug: running mpirun -np 4 xterm -e gdb ./mmm_mpi_d'
	@echo '*'
	@echo
	mpirun -np 4 xterm -e gdb ./mmm_mpi

debug_2p: mmm_mpi_d
	@echo
	@echo '*'
	@echo '* debug: running mpirun -np 2 xterm -e gdb ./mmm_mpi_d'
	@echo '*'
	@echo
	mpirun -np 2 xterm -e gdb ./mmm_mpi	

	
clean:
	rm mmm_mpi *.o 

clean_lvoc:
	rm *.gcno *.gcda *.info
	rm -r ./output
