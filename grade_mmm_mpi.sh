#!/bin/bash

set -o pipefail
#set -xv # debug

# Absolute path of this file
CWD=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)

#
# Logging helpers
#
log() {
    echo -e "${*}"
}

info() {
    log "Info: ${*}"
}
warning() {
    log "Warning: ${*}"
}
error() {
    log "Error: ${*}"
}
die() {
    error "${*}"
    exit 1
}

#
# Scoring helpers
#
TOTAL=0
ANSWERS=()

add_answer() {
    log "Score: ${1}/1.0"
    ANSWERS+=("${1},")
}

inc_total() {
    let "TOTAL++"
}

# Returns a specific line in a multi-line string
select_line() {
    # 1: string
    # 2: line to select
    echo "$(echo "${1}" | sed "${2}q;d")"
}

fail() {
    # 1: got
    # 2: expected
    log "Fail: got '${1}' but expected '${2}'"
}

pass() {
    # got
    log "Pass: ${1}"
}

compare_output_lines() {
    # 1: output
    # 2: expected
    # 3: point step
    declare -a output_lines=("${!1}")
    declare -a expect_lines=("${!2}")
    local pts_step="${3}"

    for i in ${!output_lines[*]}; do
        if [[ ${output_lines[${i}]} =~ ${expect_lines[${i}]} ]]; then
            pass "${output_lines[${i}]}"
            sub=$(bc<<<"${sub}+${pts_step}")
        else
            fail "${output_lines[${i}]}" "${expect_lines[${i}]}" ]]
        fi
    done
}

#
# Generic function for running tests
#
EXEC="mmm_mpi"
run_test() {
	#1: number of processes
	local nproc="${1}"
	shift
    #1: cli arguments
    local args=("${@}")
	echo "mpirun -n ${nproc} ${EXEC} ${args[*]}"

    # These are global variables after the test has run so clear them out now
    unset STDOUT STDERR RET

    # Create temp files for getting stdout and stderr
    local outfile=$(mktemp)
    local errfile=$(mktemp)

    # Encapsulates commands with `timeout` in case the process hangs indefinitely
    timeout 10 bash -c "mpirun -use-hwthread-cpus -n ${nproc} ${EXEC} ${args[*]}" \
        >${outfile} 2>${errfile}

    # Get the return status, stdout and stderr of the test case
    RET="${?}"
    STDOUT=$(cat "${outfile}")
    STDERR=$(cat "${errfile}")

    # Deal with the possible timeout errors
    [[ ${RET} -eq 127 ]] && die "Something is wrong (the executable might not exist)"
    [[ ${RET} -eq 124 ]] && warning "Command timed out..."

    # Clean up temp files
    rm -f "${outfile}"
    rm -f "${errfile}"
}

#
# Test cases
#
TEST_CASES=()

## Error management usage (no args)
TEST_CASES+=("err_no_arg")
err_no_arg() {
    log "\n--- Running ${FUNCNAME} ---"
    inc_total
    sub=0

    run_test "1"

    local line_array=()
    line_array+=("$(select_line "${STDERR}" "1")")
    line_array+=("Return code: '${RET}'")
    local corr_array=()
    corr_array+=("Usage: mmm_mpi N")
    corr_array+=("Return code: '1'")
    compare_output_lines line_array[@] corr_array[@] "0.5"
    add_answer "${sub}"
}

## Error management wrong matrix order ("-1")
TEST_CASES+=("err_mat_order_1")
err_mat_order_1() {
    log "\n--- Running ${FUNCNAME} ---"
    inc_total
    sub=0

    run_test "1" "-1"

    local line_array=()
    line_array+=("$(select_line "${STDERR}" "1")")
    line_array+=("Return code: '${RET}'")
    local corr_array=()
    corr_array+=("Error: wrong matrix order \(N > 0\)")
    corr_array+=("Return code: '1'")
    compare_output_lines line_array[@] corr_array[@] "0.5"
    add_answer "${sub}"
}

## Error management wrong matrix order ("titi")
TEST_CASES+=("err_mat_order_titi")
err_mat_order_titi() {
    log "\n--- Running ${FUNCNAME} ---"
    inc_total
    sub=0

    run_test "1" "titi"

    local line_array=()
    line_array+=("$(select_line "${STDERR}" "1")")
    line_array+=("Return code: '${RET}'")
    local corr_array=()
    corr_array+=("Error: wrong matrix order \(N > 0\)")
    corr_array+=("Return code: '1'")
    compare_output_lines line_array[@] corr_array[@] "0.5"
    add_answer "${sub}"
}

## 8 processes - matrix order 8
TEST_CASES+=("run_8_8")
run_8_8() {
    log "\n--- Running ${FUNCNAME} ---"
    inc_total
    sub=0

    run_test "8" "8"

    local line_array=()
    line_array+=("$(select_line "${STDOUT}" "1")")
    line_array+=("$(select_line "${STDOUT}" "2")")
    line_array+=("$(select_line "${STDOUT}" "3")")
    line_array+=("$(select_line "${STDOUT}" "4")")
    local corr_array=()
    corr_array+=("Running time: [0-9]+.[0-9]+ secs")
    corr_array+=("A: 3638197757")
    corr_array+=("B: 4038672346")
    corr_array+=("C: 2446309402")
    compare_output_lines line_array[@] corr_array[@] "0.25"
    add_answer "${sub}"
}

## 3 processes - matrix order 8
TEST_CASES+=("run_3_8")
run_3_8() {
    log "\n--- Running ${FUNCNAME} ---"
    inc_total
    sub=0

    run_test "3" "8"

    local line_array=()
    line_array+=("$(select_line "${STDOUT}" "1")")
    line_array+=("$(select_line "${STDOUT}" "2")")
    line_array+=("$(select_line "${STDOUT}" "3")")
    line_array+=("$(select_line "${STDOUT}" "4")")
    local corr_array=()
    corr_array+=("Running time: [0-9]+.[0-9]+ secs")
    corr_array+=("A: 3638197757")
    corr_array+=("B: 4038672346")
    corr_array+=("C: 2446309402")
    compare_output_lines line_array[@] corr_array[@] "0.25"
    add_answer "${sub}"
}

## 200
TEST_CASES+=("run_200")
run_200() {
    log "\n--- Running ${FUNCNAME} ---"
    inc_total
    sub=0

    run_test "8" "200"

    local line_array=()
    line_array+=("$(select_line "${STDOUT}" "1")")
    line_array+=("$(select_line "${STDOUT}" "2")")
    line_array+=("$(select_line "${STDOUT}" "3")")
    line_array+=("$(select_line "${STDOUT}" "4")")
    local corr_array=()
    corr_array+=("Running time: [0-9]+.[0-9]+ secs")
    corr_array+=("A: 530124761")
    corr_array+=("B: 1252158766")
    corr_array+=("C: 3523977945")
    compare_output_lines line_array[@] corr_array[@] "0.25"
    add_answer "${sub}"
}

## 500
TEST_CASES+=("run_500")
run_500() {
    log "\n--- Running ${FUNCNAME} ---"
    inc_total
    sub=0

    run_test "8" "500"

    local line_array=()
    line_array+=("$(select_line "${STDOUT}" "1")")
    line_array+=("$(select_line "${STDOUT}" "2")")
    line_array+=("$(select_line "${STDOUT}" "3")")
    line_array+=("$(select_line "${STDOUT}" "4")")
    local corr_array=()
    corr_array+=("Running time: [0-9]+.[0-9]+ secs")
    corr_array+=("A: 1019903034")
    corr_array+=("B: 408807171")
    corr_array+=("C: 4120804374")
    compare_output_lines line_array[@] corr_array[@] "0.25"
    add_answer "${sub}"
}

## 1000
TEST_CASES+=("run_1000")
run_1000() {
    log "\n--- Running ${FUNCNAME} ---"
    inc_total
    sub=0

    run_test "8" "1000"

    local line_array=()
    line_array+=("$(select_line "${STDOUT}" "1")")
    line_array+=("$(select_line "${STDOUT}" "2")")
    line_array+=("$(select_line "${STDOUT}" "3")")
    line_array+=("$(select_line "${STDOUT}" "4")")
    local corr_array=()
    corr_array+=("Running time: [0-9]+.[0-9]+ secs")
    corr_array+=("A: 707629981")
    corr_array+=("B: 1180118565")
    corr_array+=("C: 1795483213")
    compare_output_lines line_array[@] corr_array[@] "0.25"
    add_answer "${sub}"
}

## 2000
TEST_CASES+=("run_2000")
run_2000() {
    log "\n--- Running ${FUNCNAME} ---"
    inc_total
    sub=0

    run_test "8" "2000"

    local line_array=()
    line_array+=("$(select_line "${STDOUT}" "1")")
    line_array+=("$(select_line "${STDOUT}" "2")")
    line_array+=("$(select_line "${STDOUT}" "3")")
    line_array+=("$(select_line "${STDOUT}" "4")")
    local corr_array=()
    corr_array+=("Running time: [0-9]+.[0-9]+ secs")
    corr_array+=("A: 1149169776")
    corr_array+=("B: 2954148329")
    corr_array+=("C: 1573102286")
    compare_output_lines line_array[@] corr_array[@] "0.25"
    add_answer "${sub}"
}


#
# Main functions
#
TDIR=test_dir

clean_tdir() {
    cd ..
    rm -rf "${TDIR}"
}

make_exec() {
    # Make sure there no executable
    rm -f "${EXEC}"

    # Compile
    make > /dev/null 2>&1 ||
        die "Compilation failed"

    # Make sure that the shell executable was created
    if [[ ! -x "${EXEC}" ]]; then
        clean_tdir
        die "Can't find executable after compilation"
    fi
}

prep_tdir() {
    # Make a new testing directory
    rm -rf "${TDIR}"
    mkdir "${TDIR}" && cd "${TDIR}"

    cp "../${EXEC}" .
}

main_func() {
    # Run all the tests
    for t in "${TEST_CASES[@]}"; do
        ${t}
    done

    # Remove last comma from last answer entry
    ANSWERS[-1]=${ANSWERS[-1]%?}

    # Log the results
    log "\n\n--- Final results ---"
    log "${TOTAL} test cases were passed"
    log "${ANSWERS[*]}"
}

cd "${CWD}"
make_exec
prep_tdir
main_func
clean_tdir
