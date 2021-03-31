from typing import Union, List, Dict, Tuple, Any

from numpy import ndarray
from scipy.sparse import spmatrix

from .constraint import Constraint, ConstraintSense
from .linear_constraint import LinearConstraint
from .linear_expression import LinearExpression
from .quadratic_expression import QuadraticExpression
from .quadratic_program_element import QuadraticProgramElement

class SpecialConstraint(QuadraticProgramElement):

    def __init__(self,  quadratic_program: Any = None,
                        linear: Union[ndarray, spmatrix, List[float],
                                Dict[Union[int, str], float]] = None,
                        linear_sense: Union[str, ConstraintSense] = '<=',
                        linear_rhs: float = 0.0,
                        penalty_constant: float =0.0,
                        penalty_linear_coeff: Union[ndarray, spmatrix, List[float],
                                        Dict[Union[int, str], float]] = None,
                        penalty_quadratic_coeff:  Union[ndarray, spmatrix, List[List[float]],
                                            Dict[Tuple[Union[int, str], Union[int, str]], float]] = None,) -> None:
        self._quadratic_program = quadratic_program
        super().__init__(self._quadratic_program)
        #self._name = name
        self._linear_constraint = LinearConstraint(self._quadratic_program, None, linear, Constraint.Sense.convert(linear_sense), linear_rhs)
        #penalty
        self.penalty_constant = penalty_constant
        self.penalty_linear_expression = LinearExpression(self._quadratic_program, penalty_linear_coeff)
        self.penalty_quadratic_expression = QuadraticExpression(self._quadratic_program, penalty_quadratic_coeff)

    def is_special_constraint(self, linear_constraint: LinearConstraint) -> bool: 
        if (self._linear_constraint.linear.to_dict() == linear_constraint.linear.to_dict()
            and self._linear_constraint.sense == linear_constraint.sense
            and self._linear_constraint.rhs == linear_constraint.rhs):
            return True
        return False

