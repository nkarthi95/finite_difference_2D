export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# 2D decomposition, OMP #
make clean
make USE_SENDV_1D=0 USE_OMP=1 USE_OLDMPI=0

OUTPUT_FILE="profiling-non_blocking-hybrid-2D.txt"
rm -f $OUTPUT_FILE  # Remove old output file if exists

for i in $(seq 1 ${1});
do
    mpirun -n 2 --bind-to core --map-by socket heat_solver >> $OUTPUT_FILE
    echo "----------" >> $OUTPUT_FILE
done

# 2D decomposition, non OMP #
make clean
make USE_SENDV_1D=0 USE_OMP=0 USE_OLDMPI=0

OUTPUT_FILE="profiling-non_blocking-non_hybrid-2D.txt"
rm -f $OUTPUT_FILE  # Remove old output file if exists

for i in $(seq 1 ${1});
do
    mpirun -n 2 --bind-to core --map-by socket heat_solver >> $OUTPUT_FILE
    echo "----------" >> $OUTPUT_FILE
done

# Naive decomposition, non blocking comms, OMP #
make clean
make USE_SENDV_1D=0 USE_OMP=1 USE_OLDMPI=1

OUTPUT_FILE="profiling-non_blocking-hybrid-1D.txt"
rm -f $OUTPUT_FILE  # Remove old output file if exists

for i in $(seq 1 ${1});
do
    mpirun -n 2 --bind-to core --map-by socket heat_solver >> $OUTPUT_FILE
    echo "----------" >> $OUTPUT_FILE
done


# Naive decomposition, non blocking comms#
make clean
make USE_SENDV_1D=0 USE_OMP=0 USE_OLDMPI=1

OUTPUT_FILE="profiling-non_blocking-non_hybrid-1D.txt"
rm -f $OUTPUT_FILE  # Remove old output file if exists

for i in $(seq 1 ${1});
do
    mpirun -n 2 --bind-to core --map-by socket heat_solver >> $OUTPUT_FILE
    echo "----------" >> $OUTPUT_FILE
done

# Naive decomposition, blocking comms, OMP#
make clean
make USE_SENDV_1D=1 USE_OMP=1 USE_OLDMPI=1

OUTPUT_FILE="profiling-blocking-hybrid-1D.txt"
rm -f $OUTPUT_FILE  # Remove old output file if exists

for i in $(seq 1 ${1});
do
    mpirun -n 2 --bind-to core --map-by socket heat_solver >> $OUTPUT_FILE
    echo "----------" >> $OUTPUT_FILE
done

make clean

# Naive decomposition, blocking comms#
make clean
make USE_SENDV_1D=1 USE_OMP=0 USE_OLDMPI=1

OUTPUT_FILE="profiling-blocking-non_hybrid-1D.txt"
rm -f $OUTPUT_FILE  # Remove old output file if exists

for i in $(seq 1 ${1});
do
    mpirun -n 2 --bind-to core --map-by socket heat_solver >> $OUTPUT_FILE
    echo "----------" >> $OUTPUT_FILE
done

make clean  