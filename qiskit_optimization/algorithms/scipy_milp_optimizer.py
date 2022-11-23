"""The MILP(Scipy) optimizer wrapped to be used within Qiskit's optimization module."""
import qiskit_optimization.optionals as _optionals
from ..exceptions import QiskitOptimizationError
from ..problems.quadratic_program import QuadraticProgram
from .optimization_algorithm import OptimizationAlgorithm, OptimizationResult
from packaging import version
import numpy as np
import scipy.version
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix

class ScipyMilpOptimizer(OptimizationAlgorithm):
    
    def __init__(self, disp: bool = False) -> None:
        """Initializes the ScipyMILPOptimizer.

        Args:
            disp: Whether to print MILP output or not.
        """
        self._disp = disp

    @staticmethod
    def is_scipy_updated():
        """Returns True if gurobi is installed"""
        return version.parse("1.9.0") < version.parse(scipy.version.version) ############################## check version of scipy so we have MILP solver


    @property
    def disp(self) -> bool:
        """Returns the display setting.

        Returns:
            Whether to print MILP information or not.
        """
        return self._disp

    @disp.setter
    def disp(self, disp: bool):
        """Set the display setting.
        Args:
            disp: The display setting.
        """
        self._disp = disp

    # pylint:disable=unused-argument
    def get_compatibility_msg(self, problem: QuadraticProgram) -> str:
        """Checks whether a given problem can be solved with this optimizer.
        The scipy.milp only solves linear problems. Check if the problem is linear by objective function and constraints.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            True or False
        """
        msg = ""
        if problem.objective.quadratic.to_dict()!={}:
            msg += "Quadratic objective function is not supported!"
        if problem.quadratic_constraints:
            msg += "Quadratic constraints are not supported!"
        return msg
    
    def _generate_problem(problem: QuadraticProgram):
        
        ## Obtain sense of objective function (+1 for minimization and -1 for maximization)
        sense = problem.objective.sense.value

        ## Obtain coefficient of objective function
        objective = problem.objective.linear.to_array() * sense  

        ## Initialize constraints matrix
        constraints_matrix = csc_matrix(len(problem.linear_constraints), len(problem.variables))

        ## Initialize constraint value
        left_constraint = np.array([])
        right_constraint = np.array([])

        for i, constraint in enumerate(problem.linear_constraints):
            constraint_dict = constraint.linear.to_dict()
            for var in constraint_dict:
                constraints_matrix[i,var]= constraint_dict[var]
            
            left_constraint = np.append(left_constraint, [constraint.rhs if constraint.sense.value==1 else -np.inf])
            right_constraint = np.append(right_constraint, [constraint.rhs if constraint.sense.value==0 else np.inf])
        
        constraints = LinearConstraint(constraints_matrix, left_constraint, right_constraint)
        integrality = np.array([ 1 if variable.vartype.value==2 or variable.vartype.value==1 else 0 for variable in problem.variables])
        lower_bounds = np.array([variable.lowerbound for variable in problem.variables])
        upper_bounds = np.array([variable.upperbound for variable in problem.variables])
        bounds = Bounds(lower_bounds,upper_bounds)

        return objective, constraints, integrality, bounds, sense


    def solve(self, problem: QuadraticProgram) -> OptimizationResult:
        """Tries to solves the given problem using the optimizer.

        Runs the optimizer to try to solve the optimization problem. If problem is not convex,
        this optimizer may raise an exception due to incompatibility, depending on the settings.

        Args:
            problem: The problem to be solved.

        Returns:
            The result of the optimizer applied to the problem.

        Raises:
            QiskitOptimizationError: If the problem is incompatible with the optimizer.
        """
        # pylint: disable=import-error

        objective, constraints, integrality, bounds, sense = _generate_problem(problem)
        opt_result = milp(objective, constraints, integrality, bounds)


        
        # create results
        result = OptimizationResult(
            x=opt_result.x,
            fval=opt_result.fun*sense,
            variables=problem.variables,
            #status=self._get_feasibility_status(problem, model.X),
            status=opt_result.status,
            raw_results=opt_result
        )

        # return solution
        return result
