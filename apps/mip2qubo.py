import sys
import argparse

from qiskit_optimization import QuadraticProgram

###
# Read a larger LP file and convert it to other formats
###

options = argparse.ArgumentParser("mip2qubo")
options.add_argument("problem")

args = options.parse_args()

# read LP file in Qiskit format
qp = QuadraticProgram()
qp.read_from_mps_file(args.problem)
qp.write_to_lp_file(str(args.problem).removesuffix(".mps.gz"))
