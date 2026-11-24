import numpy as np
import torch
import logging
from sympy import *
from dataclasses import dataclass
from pysdd.sdd import SddManager, SddNode
from pysdd.util import sdd_to_dot
from typing import Callable, Dict, List, Tuple, Optional, Set, Optional
from .circuit_solvers import SumProductRegistry

logger = logging.getLogger(__name__)

@dataclass
class ProgramCircuit:
    """Helper class for representating circuit and performing circuit operations.

    Attributes:
        mgr: SDDManager for the circuit.
        circuit: The SDDNode representing the circuit.
        literal_map: A map from literals to program symbol names.
        weights: The weights underlying the nodes in circuit.
        bool_set: The set of unweighted Boolean variables.

    Methods:
        dot: Translates circuit to dot representation.
        inference: Runs an inference over the circuit based on the
            the specified `mode` of inference.
    """
    mgr: SddManager
    circuit: SddNode
    literal_map: dict
    weight_map: dict
    weights: torch.Tensor
    bool_set: Set[int]
    return_vars: Set[str]

    def __post_init__(self):
        self._reverse_map = {v: k for k, v in self.literal_map.items()}

    def dot(self) -> str:
        """Maps the circuit to graphviz graph.

        Returns:
            String representation of graphviz graph.

        """
        return sdd_to_dot(self.circuit, litnamemap=self.literal_map)

    def inference(
        self, 
        mode="prob", 
        marginals: Optional[set] = set(), 
        has_evidence: Optional[bool] = False, 
        compute_grad: Optional[bool] = False, 
        numeric_funcs: Optional[Dict[str, Tuple[Callable, List[str]]]] = None, 
        return_pytorch: Optional[bool] = False,
        categoricals: Optional[Set[str]] = None,
        weight_id: Optional[int] = None,
    ):
        """Compute an inference over the circuit based on the provided
            `mode` of reasoning.

        Arguments:
            mode: The particular type of reasoning and inference to perform.
            marginals: The set of marginals to compute

        """
        return reason(
            self, 
            mode, 
            marginals, 
            has_evidence=has_evidence, 
            compute_grad=compute_grad, 
            numeric_funcs=numeric_funcs,
            return_pytorch=return_pytorch,
            categoricals=categoricals,
            weight_id=weight_id,
        )
    
class WeightGraph:
    """Computation graph over weight variables, built from numeric_funcs.
    
    Leaf nodes are free parameters (looked up from the weight tensor).
    Derived nodes are computed recursively from their deps, with memoization
    to ensure each node is evaluated exactly once regardless of query order.
    """
    def __init__(self, numeric_funcs):
        self.funcs = numeric_funcs  # {name: (fn, [dep_name | float])}

    def evaluate(self, var_name, weights, weight_map, reverse_map, memo):
        if isinstance(var_name, str) and var_name in memo:
            return memo[var_name]  # already computed (handles diamond deps too)
        elif isinstance(var_name, tuple):
            func, deps, index = var_name
            resolved = [
                self.evaluate(d, weights, weight_map, reverse_map, memo)
                if isinstance(d, (str, tuple))
                else torch.tensor(d, dtype=weights.dtype, device=weights.device)
                for d in deps
            ]
            val = func(*resolved)
        elif var_name in self.funcs:
            fn, deps, index = self.funcs[var_name]
            resolved = [
                self.evaluate(d, weights, weight_map, reverse_map, memo)
                if isinstance(d, (str, tuple))
                else torch.tensor(d, dtype=weights.dtype, device=weights.device)
                for d in deps
            ]
            val = fn(*resolved)
            if index != -1: 
                val = val[index]
        else: 
            # leaf: free parameter
            val = weights[weight_map[reverse_map[var_name]]]
        
        if isinstance(var_name, str):
            memo[var_name] = val
        return val

    def compute_all(self, literal_map, weights, weight_map, reverse_map, reverse_weight_map):
        memo = {} 

        for i in range(len(weights)):
            lit_id = reverse_weight_map[i]
            var_name = literal_map[lit_id]
            self.evaluate(var_name, weights, weight_map, reverse_map, memo)
        return memo
    

class SolverWeights(torch.nn.Module):
    """Pytorch module for storing solver weights. Helper for also computing gradients    
    and reporting them back.
    
    Attributes: 
        weights: The underlying solver weights. 
    """

    def __init__(self, solver_weights, numeric_funcs: Optional[Dict[str, Tuple[Callable, List[str]]]] = None):
        super().__init__()
        self.weights = torch.nn.Parameter(solver_weights)
        self.numeric_funcs = numeric_funcs

    def forward(self, circuit): 
        """Some parameters might depend on other parameters. In order to manage this, we 
        need to recompute the `weight` so that such computations are plugged into the 
        computation graph. 

        Arguments: 
            circuit: The underlying circuit with information about variables and their 
                place in the weight tensor. 
            numeric_funcs: A specification of those extra functions (if provided). 

        Returns: 
            Either the original weight tensor (if no numeric functions are provided), or an 
                updated weight tensor.         
        """
        if self.numeric_funcs is None:
            return self.weights
                
        graph = WeightGraph(self.numeric_funcs)
        memo = {} 
        reverse_weight_map = {v: k for k, v in circuit.weight_map.items()}
        memo = graph.compute_all(
            circuit.literal_map,
            self.weights, 
            circuit.weight_map, 
            circuit._reverse_map,
            reverse_weight_map,
        )
        return torch.stack(
            [memo[circuit.literal_map[reverse_weight_map[i]]] for i in range(len(self.weights))]
        )
        
def reason(
    circuit, 
    mode, 
    marginals: Optional[set] = set(), 
    has_evidence: Optional[bool] = False, 
    compute_grad: Optional[bool] = False, 
    numeric_funcs: Optional[Dict[str, Tuple[Callable, List[str]]]] = None,
    return_pytorch: Optional[bool] = False,
    categoricals: Optional[Set[str]] = None,
    weight_id: Optional[int] = None,
):
    """Performs the underlying reasoning.

    Arguments:
        program: The underlying program and its circuit, weights, etc.
        mode: The particular mode of reasoning.
        marginals: The marginals to compute.

    Raises:
        ValueError whenever the solver mode is not recognized
            (currently implements: `prob`).

    """
    ## initialize solver instance 
    solver = SumProductRegistry.build_instance(mode)

    ### get out marginal ids
    marginal_ids = set()
    for literal in marginals:
        if literal not in circuit._reverse_map:
            logger.warning(f"WARNING: unknown marginal id: {literal}, skipping...")
            continue
        marginal_ids.add(circuit._reverse_map[literal])
    
    ## get out categorical ids
    categorical_ids = set()
    for literal in categoricals or set():
        if literal not in circuit._reverse_map:
            logger.warning(f"WARNING: unknown categorical id: {literal}, skipping...")
            continue
        categorical_ids.add(-circuit._reverse_map[literal])

    ### change computation to point to other variables if specified.  
    if weight_id is not None:
        pass 

    ### preprocess weights, turn on gradients if necessary
    weight_helper = SolverWeights(circuit.weights, numeric_funcs)    
    weights = weight_helper(circuit)
        
    ### perform evaluation
    output =  solver.evaluate(
        circuit.circuit,
        circuit.mgr,
        weights=weights, 
        weight_map=circuit.weight_map,
        marginals=marginal_ids,
        literal_map=circuit.literal_map,
        has_evidence=has_evidence,
        categoricals=categorical_ids,
    )

    if compute_grad: 
        try: 
            ## compute grad over `output.out` above,  
            grad_wrt_weights = torch.autograd.grad(
                outputs=output.out,
                inputs=weight_helper.weights,
                allow_unused=True,
                retain_graph=False,
                create_graph=False,
            )[0]
            grad_list = []
            weight_grads = grad_wrt_weights.detach().tolist()
            for sdd_id, var_id in circuit.literal_map.items():
                if (sdd_id < 0) or (sdd_id not in circuit.weight_map):
                    continue
                grad_list.append((var_id, weight_grads[circuit.weight_map[sdd_id]]))
            output.param_grads = grad_list
        except: 
            logger.warning("WARNING: gradient computation failed, skipping...")
            output.param_grads = [] 

    if return_pytorch is False: 
        output.out = output.out.detach().tolist() if isinstance(output.out, torch.Tensor) else output.out
        output.Z = output.Z.detach().tolist() if isinstance(output.Z, torch.Tensor) else output.Z

    return output 