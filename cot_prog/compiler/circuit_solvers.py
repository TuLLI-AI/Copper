import operator
import torch
from dataclasses import dataclass
from pysdd.sdd import SddManager, SddNode
from pysdd.iterator import SddIterator
from typing import Callable, Optional, Tuple, List, Union, Set, Dict, ClassVar, Any

from ..base import UtilBase
from ..register import Registry 

__all__ = [
    "Prob",
    "MPE",
    "Count",
    "SolverRegistry",
]

class SumProductRegistry(Registry):
    """Registry for the different kinds of solvers """
    
    def _verify_entry(self, register_cls):
        """Verifies that registered semirings have all required 
        semiring attributes. 

        Raises: 
            ValueError whenever an implementation is missing an attribute. 
        """
        for attr in [
            "Sum", "Product", "Negation", "Division", "One", "Zero"
        ]:
            if not hasattr(register_cls, attr) or getattr(register_cls,attr) is None:
                raise ValueError(f"Implementation missing attribute: {attr}")
                
    
class GeneralizedSumProduct(UtilBase):
    """Base class for defining different `GeneralizedSumProduct`
    problems. Formally: implements a particular semiring in the
    Algebraic Model Counting framework (AMC, https://arxiv.org/abs/1211.4475).

    Attributes:
        sum: The implementation of the summation or addition binary operator.
        product: The implementation of the product or multiplication binary operator.
        negation: The implementatin of the negation operator.
        zero: The implementation of the indentity element for addition.
        one: The implementation of the identity element for multiplication.

    see: https://github.com/ML-KULeuven/problog/blob/59b107364b4e4c4c399aa84d53cd054e5dde1d6e/problog/evaluator.py#L35
    """
    Sum: ClassVar[Callable]
    Product : ClassVar[Callable]
    Negation: ClassVar[Callable]
    Division: ClassVar[Callable]
    One: ClassVar[Any]
    Zero: ClassVar[Any]

    def negation(self, x):
        """Main negation operator.

        Returns:
            The particular implementation of negation.
        """
        raise NotImplementedError

    def sum(self, x, y):
        """Main sum or addition operator.
            addition.

        Returns:
            The particular function implementation for summation.

        """
        raise NotImplementedError

    def product(self, x, y):
        """Main product/multiplication operator.

        Returns:
            The particular function implementation for product.

        """
        raise NotImplementedError

    @property
    def zero(self):
        """Returns the identity element for summation."""
        raise NotImplementedError

    @property
    def one(self) -> Callable:
        """Returns the identity element for product"""
        return NotImplementedError

    @property
    def true(self):
        return self.one

    @property
    def requires_smoothing(self) -> bool:
        """Returns Boolean indicating whether this semiring requires smoothness when 
        computing the weighted model counts."""
        return False 

    @property
    def false(self):
        return self.zero

    def assignment(self, identifier):
        """Assigns a value to a particular literal node

        Arguments:
            identifier: The identifier associated with the literal.

        """
        raise NotImplementedError


TupleType = Tuple[Union[float, torch.tensor], Union[float, torch.tensor, List]]

@dataclass
class ReasoningOutput:
    """Helper class for returning circuit reasoner output.

    Attributes:
        Z: (optional) normalization constant
        out: output score provided by solver.
        marginals: (optional) list of marginals used in computation.
        param_grads: (optional) Gradients of weight parameters under computation.

    """
    out: float
    marginals: Set[str]
    param_grads: List[float]
    assignments: List[str]
    Z: Optional[float] = 1.0


class TupleSumProduct(GeneralizedSumProduct):
    """Sum product problems with ordinary addition and multiplication. Uses
    tuples to keep track of marginals, evidence, etc.

    Class Attributes:
        Sum: The main sum operator (default: ordinary addition).
        Product: The main product operator (default: ordinary mult).
        Negation: The main negation operator.
        One: The identity element for product
        Zero: The identity element for sum

    Attributes:
        sum: The implementation of the summation or addition binary operator.
        product: The implementation of the product or multiplication binary operator.
        negation: The implementatin of the negation operator.
        zero: The implementation of the indentity element for addition.
        one: The implementation of the identity element for multiplication.

    """

    Sum = staticmethod(operator.add)
    Product = staticmethod(operator.mul)
    Negation = staticmethod(lambda p: 1.0 - p)
    Division = staticmethod(lambda p, Z: p / Z)
    One = 1.0
    Zero = 0.0

    @property
    def zero(self) -> TupleType:
        """Returns the identity element for summation."""
        return (self.Zero, self.Zero)

    @property
    def one(self) -> TupleType:
        """Returns the identity element for product"""
        return (self.One, self.One)

    def negation(self, x: float) -> float:
        """Implementation of the negation operator

        Arguments:
            x: The weight to negate (assumes a probability).

        """
        return self.Negation(x)

    def sum(self, x: TupleType, y: TupleType) -> TupleType:
        """Main sum or addition operator. Ordinary addition.

        Arguments:
            x: The first tuple argument.
            y: The second tuple argument.
        Returns:
            The particular function implementation for summation.

        """
        return (self.Sum(x[0], y[0]), self.Sum(x[1], y[1]))

    def product(self, x: TupleType, y: TupleType) -> TupleType:
        """Main product/multiplication operator. Ordinary multiplication.

        Arguments:
            x: The first tuple argument.
            y: The second tuple argument.
        Returns:
            The particular function implementation for product.

        """
        return (self.Product(x[0], y[0]), self.Product(x[1], y[1]))

    def assignment(self, identifier: int) -> TupleType:
        """Generic assignment function that returns literal weights and
        raw Booleans if no weights are provided or. 

        Arguments:
            identifier: The identifier associated with the literal.

        """
        norm_id = abs(identifier)  #identifier if identifier > 0 else -identifier
        atom_weight = torch.tensor(1.0)
        if norm_id in self.weight_map and identifier not in self.categoricals:
            atom_weight = self.weights[self.weight_map[norm_id]]
            if identifier < 0:
                atom_weight = self.negation(atom_weight)
        second = atom_weight.clone()
        if identifier in self.marginals:
            second = torch.tensor(1.0) #<--- set device, use semiring 1
        elif -identifier in self.marginals:
            second = torch.tensor(0.0) #<000 set device, use semiring 0.

        return (second, atom_weight)

    def _format_output(self, output, **kwargs) -> ReasoningOutput:
        """Format the final output of the solver.

        Arguments:
            output: The raw output from `evaluate`.
            **kwargs: Other possible outputs.
        Returns:
            A formatting version of ReasoningOutput

        """
        raise NotImplementedError("Formatting now implemented for this solver")

    @staticmethod
    def _get_marginals_assignments(
        *,
        literal_map: Dict,
        marginals: Optional[Set] = None,
        assignments: Optional[List] = None,
    ) -> Tuple[Set[int],List[int]]:
        """Method for computing marginal assignmnents. 

        Arguments: 
            marginals: The set of marginals in the computation. 
            assignments: The list of variable assignments. 
            literal_map: Mapping between literal ids and their circuit ids.

        """
        if marginals is None:
            marginals = set()
        if assignments is None:
            assignments = []
            
        marginals_out = set()
        for m in marginals:
            if m in literal_map:
                marginals_out.add(literal_map[m])

        assignments_out = []
        for a in assignments:
            if a in literal_map:
                assignments_out.append(literal_map[a])
            else:
                assignments_out.append(a)

        return (marginals_out, assignments_out)

    @staticmethod
    def _get_grad_map(
        quantity: torch.tensor,
        weights: torch.tensor,
        literal_map: Optional[dict] = None,
        weight_map: Optional[dict] = None,
    ) -> List[Tuple[int, float]]:
        """Get gradient information from a set of weights

        Arguments:
            quantity: A pytorch computation used for getting the getting gradients.
            weights: The underlying weights of the model.
            literal_map: A map between original variable names and literals inside of circuit.
            weight_map: A map between circuits literals and positions in weights.

        """
        grad_list = []
        if literal_map is None:
            literal_map = {}
        if weight_map is None:
            weight_map = {}
        
        if literal_map is None or weight_map is None or weights is None:
            return grad_list
        if not isinstance(weights, torch.Tensor) or not weights.requires_grad:
            return grad_list
        if quantity.ndim != 0:
            quantity = quantity.sum()
        if not quantity.requires_grad:
            return grad_list 

        grad_wrt_weights = torch.autograd.grad(
            outputs=quantity,
            inputs=weights,
            allow_unused=True,
            retain_graph=False,
            create_graph=False,
        )[0]
        if grad_wrt_weights is None:
            return grad_list

        weight_grads = grad_wrt_weights.detach().tolist()
        for sdd_id, var_id in literal_map.items():
            if (sdd_id < 0) or (sdd_id not in weight_map):
                continue
            grad_list.append((var_id, weight_grads[weight_map[sdd_id]]))
            
        return grad_list

    def evaluate(
        self,
        circuit: SddNode,
        manager: SddManager,
        weights: torch.tensor,
        weight_map: Dict[int, int],
        marginals: Optional[set] = None,
        literal_map: Optional[Dict] = None,
        compute_gradients: Optional[bool] = False,
        smoothing_to_root: Optional[bool] = False,
        has_evidence: Optional[bool] = False,
        categoricals: Optional[Set[str]] = None,
    ) -> ReasoningOutput:
        """High-level function for evaluating and computing over a circuit.

        Arguments:
            circuit: The input SDD circuit.
            manager: The underlying SDD manager that contains the vtree.
            weights: The underlying weight parameters for reasoning.
            marginals: The list of marginals.
            categoricals: List of categorical variables. 

        """
        if marginals is None:
            marginals = set()
        if literal_map is None:
            literal_map = {}
        
        self.weights = weights
        self.weight_map = weight_map
        self.marginals = marginals
        self.categoricals = categoricals
        self.manager = manager
        self.literal_map = literal_map
        self._eval_cache = {}
        self._smooth_cache = {}

        smoothing_to_root = smoothing_to_root
        sdd_iterator = SddIterator(self.manager, smooth_to_root=smoothing_to_root)
        output = sdd_iterator.depth_first(circuit, self._get_wmc_func())

        normalize = True if (marginals or has_evidence) else False
        return self._format_output(
            output, normalize=normalize, compute_gradients=compute_gradients,
        )
    
    def _get_wmc_func(self, perform_smoothing: bool = True) -> Callable:
        """Defines the WMC function to use for the SddIterator, which implements smoothing. Modeled
        after the problog implementation. 
        
        see: https://github.com/ML-KULeuven/problog/blob/d5cece5da76ce58151eeb3d554e379b65e4d8bf5/problog/sdd_formula.py#L508

        Arguments: 
            perform_smoothing: Switch to determine if smoothing is used. Note that this 
                gets overrided if the semiring requires smoothing (default: false). 
        """
        smooth_flag = perform_smoothing or self.requires_smoothing 
        
        def func_wmc(node: SddNode, rvalues, expected_prime_vars: Set[int], expected_sub_vars: Set[int]):
            """The SddIterator compatible depth-first WMC subroutine that is defiend for
            our semiring implementations

            Arguments: 
                node: The incoming sdd node. 
                rvalues: The decision node values (if provided) 
                expected_prime_vars: The full variables needed for smoothing. 
                expected_subvars: The extra variables needed for smoothing. 

            """
            if rvalues is None:

                if node.is_true():
                    val = self.true
                    if smooth_flag:
                        missing_literals = (
                            expected_prime_vars
                            if expected_prime_vars is not None
                            else set()
                        )
                        missing_literals |= (
                            expected_sub_vars
                            if expected_sub_vars is not None
                            else set()
                        )
                        for l in missing_literals:
                            missing_combined_weight = self.sum(self.assignment(+l),self.assignment(-l))
                            val = self.product(val, missing_combined_weight)
                            
                    return val 

                elif node.is_false():
                    return self.false

                elif node.is_literal():
                    val = self.assignment(node.literal)

                    if smooth_flag:
                        lit_scope = {abs(node.literal)}
                        if expected_prime_vars is not None:
                            missing_literals = expected_prime_vars.difference(lit_scope)
                        else:
                            missing_literals = set()
                        if expected_sub_vars is not None:
                            missing_literals |= expected_sub_vars.difference(lit_scope)

                        for l in missing_literals:
                            missing_combined_weight = self.sum(self.assignment(+l),self.assignment(-l))
                            val = self.product(val, missing_combined_weight)
                    
                    return val
                    
            else:
                if node is not None and not node.is_decision():
                    raise ValueError(r'Unknown node type: {node}')

                out = self.zero
                for prime_weight, sub_weight, prime_vars, sub_vars in rvalues:
                    branch_weight = self.product(prime_weight, sub_weight)

                    if smooth_flag:
                        missing_literals = expected_prime_vars.difference(
                            prime_vars
                        ) | expected_sub_vars.difference(sub_vars)
                        for l in missing_literals:
                            missing_combined_weight = self.sum(self.assignment(+l),self.assignment(-l))
                            branch_weight = self.product(branch_weight, missing_combined_weight)
                        
                    out = self.sum(out, branch_weight)
                
                return out 
        
        return func_wmc

@SumProductRegistry(name = "Prob")
class Prob(TupleSumProduct):
    """Implementation of the Probability semiring and sum product problem.

    Class Attributes:
        Sum: The main sum operator (default: ordinary addition).
        Product: The main product operator (default: ordinary mult).
        Negation: The main negation operator.
        One: The identity element for product
        Zero: The identity element for sum

    """

    Sum = staticmethod(torch.add)
    Product = staticmethod(torch.mul)
    Negation = staticmethod(lambda p: 1.0 - p)
    One = 1.0
    Zero = 0.0

    def _format_output(
            self, output, compute_gradients: Optional[bool] = False, normalize: Optional[bool] = False
        ) -> ReasoningOutput:
        """Format the final output of the solver.

        Arguments:
            output: The raw output from `evaluate`.
            compute_gradients: Switch for computing gradients of the weight parameters (default=False) 
            normalize: Switch for normalizing the output (default=False)
        Returns:
            A formatting version of ReasoningOutput. 
        """
        grad_list = []
        marginals = []
        prob, Z = output

        marginals, _ = self._get_marginals_assignments(
            marginals=self.marginals, literal_map=self.literal_map
        )
        if normalize: 
            prob = self.Division(prob, Z)

        # ## compute gradients if necessary
        # if prob != 0.0 and compute_gradients: 
        #     grad_list = self._get_grad_map(
        #         prob, self.weights, self.literal_map, self.weight_map
        #     )
        
        #prob = prob #.detach().tolist()
        #Z = Z #.detach().tolist()

        output = ReasoningOutput(
            out=prob,
            Z=Z,
            param_grads=grad_list,
            marginals=marginals,
            assignments=[],
        )
        return output

@SumProductRegistry(name = "Count")
class Count(Prob):
    """Implementation of the count semiring for model counting. 
    
    Class Attributes:
        Sum: The main sum operator (default: sum).
        Product: The main product operator (default: ordinary mult).
        Negation: The main negation operator.
        One: The identity element for product
        Zero: The identity element for sum

    """
    Sum = staticmethod(operator.add)
    Product = staticmethod(operator.mul)
    Negation = staticmethod(lambda p: 1.0)
    One = 1.0
    Zero = 0.0

    @property
    def requires_smoothing(self) -> bool:
        """Returns Boolean indicating whether this semiring requires smoothness when 
        computing the weighted model counts."""
        return True 

    def assignment(self, identifier: int) -> TupleType:
        norm_id = identifier if identifier > 0 else -identifier
        atom_weight = torch.tensor(1.0)
        second = torch.tensor(1.0)
        
        return (second, atom_weight)

    def _format_output(
            self, output, compute_gradients: Optional[bool] = False, normalize: Optional[bool] = False
        ) -> ReasoningOutput:
        """Format the final output of the solver.

        Arguments:
            output: The raw output from `evaluate`.
        Returns:
            A formatting version of ReasoningOutput

        """
        count, Z = output
        count = count.detach().tolist()
        Z = Z.detach().tolist()

        output = ReasoningOutput(
            out=count,
            Z=Z,
            param_grads=[],
            marginals=[],
            assignments=[],
        )
        return output

#@SolverRegistry(name = "MPE")
class MPE(TupleSumProduct):
    """Implements the `Most Probable Explanation` or Viterbi inference.

    Class Attributes:
        Sum: The main sum operator (default: max).
        Product: The main product operator (default: ordinary mult).
        Negation: The main negation operator.
        One: The identity element for product
        Zero: The identity element for sum

    """
    Sum = staticmethod(lambda x, y: x if x > y else y)
    Product = staticmethod(torch.mul)
    Negation = staticmethod(lambda p: 1.0 - p)
    One =  1.0
    Zero = 1.0 #(0.0, [])
    
    @property
    def requires_smoothing(self) -> bool:
        """Returns Boolean indicating whether this semiring requires smoothness when 
        computing the weighted model counts."""
        return True

    @property
    def zero(self) -> TupleType:
        """Returns the identity element for summation."""
        return (self.Zero, [])

    @property
    def one(self) -> TupleType:
        """Returns the identity element for product"""
        return (self.One, [])

    def sum(self, x: TupleType, y: TupleType) -> TupleType:
        """Main product/multiplication operator. Ordinary multiplication
        on left side plus list addition on right side.

        Arguments:
            x: The first tuple argument.
            y: The second tuple argument.
        Returns:
            The particular function implementation for product.
        """
        return self.Sum(x, y)

    def product(self, x: TupleType, y: TupleType) -> TupleType:
        """Main product/multiplication operator. Ordinary multiplication
        on left side plus list addition on right side.

        Arguments:
            x: The first tuple argument.
            y: The second tuple argument.
        Returns:
            The particular function implementation for product.
        """
        return (self.Product(x[0], y[0]), x[1] + y[1])

    def assignment(self, identifier):
        """Assigns a value to a particular literal node.

        Arguments:
            identifier: The identifier associated with the literal.
        """
        norm_id = identifier if identifier > 0 else -identifier
        atom_weight = torch.tensor(1.0)
        if norm_id in self.weight_map:
            atom_weight = self.weights[self.weight_map[norm_id]]
            if identifier < 0:
                atom_weight = self.negation(atom_weight)

        second = atom_weight
        if identifier in self.marginals:
            second = torch.tensor(1.0)
        elif -identifier in self.marginals:
            second = torch.tensor(0.0)

        return (second, [identifier])

    def _format_output(
            self, output, compute_gradients: Optional[bool] = False, normalize: Optional[bool] = False
        ) -> ReasoningOutput:
        """Format the final output of the solver.

        Arguments:
            output: The raw output from `evaluate`.
        Returns:
            A formatting version of ReasoningOutput

        """
        prob, assignments = output
        grad_list = []

        if compute_gradients: 
            grad_list = self._get_grad_map(
                -torch.log(prob), self.weights, self.literal_map, self.weight_map
            )

        marginals, assignments = self._get_marginals_assignments(
            marginals=self.marginals,
            assignments=assignments,
            literal_map=self.literal_map,
        )

        prob = prob.detach().tolist()

        output = ReasoningOutput(
            out=prob,
            Z=1.0,
            param_grads=grad_list,
            marginals=marginals,
            assignments=assignments,
        )
        return output

# class LogMPE(MPE):
#     pass 
# class Fuzzy(Prob):
#     """Implements the fuzzy (Goedel) semiring. """
#     Sum = staticmethod(lambda x, y: max(x, y))
#     Product = staticmethod(lambda x, y: min(x, y))
#     Negation = staticmethod(lambda p: 1.0 - p)
#     One =  1.0 #(1.0, 1.0)
#     Zero = 0.0 #(0.0, 0.0)
# class Entropy(TupleSumProduct):
#     """Implements the Entropy semiring. 
#     Class Attributes:
#         Sum: The main sum operator (default: max).
#         Product: The main product operator (default: ordinary mult).
#         Negation: The main negation operator.
#         One: The identity element for product
#         Zero: The identity element for sum
#     """
#     Sum = staticmethod(torch.logaddexp)
#     Product = staticmethod(torch.add)
#     Negation = staticmethod(lambda p: 1.0 - p)
#     Division = staticmethod(lambda p, Z: P - Z)
#     One = (1.0, 1.0)
#     Zero = (0.0, 0.0)


# if __name__ == "__main__":
#     from pysdd.sdd import SddManager
#     mgr = SddManager(var_count=2)
#     a, b = mgr.vars
#     f = a #(~a & ~b) | a
#     weights = torch.tensor([0.92, 0.7])
#     weight_map = {1: 0, 2: 1}
#     solver = Prob()
#     out = solver.evaluate(
#         f,
#         mgr,
#         weights=weights,
#         weight_map=weight_map,
#     )
#     print(out)
    #SddIterator(f.manager)
    # solver = MPE()
    # out = solver.evaluate(
    #     f,
    #     mgr,
    #     weights=weights,
    #     weight_map=weight_map,
    # )

    # op = 0.0
    # for model in f.models():
    #     lprobs = []
    #     lp = 1.0
    #     for v, val in model.items():
    #         p = weights[weight_map[v]]
    #         p = p if val == 1 else (1 - p)
    #         lp = lp * p

    #     op += lp

    # # print(op)
    # print(out)
