import sympy
import ast
import torch
import astor
import textwrap
import warnings
from sympy import *
from dataclasses import dataclass
from collections import defaultdict
from typing import Union, Callable, Dict, Tuple, List, Optional, Set, Any
from .compiler import (
    ProgramToSemantics,
    VarCollector,
    CoTProgramTransformer,
    ProbRefRewriter,
    PROBABILITY_VARIABLES,
    PROBABILITY_FUNCTIONS,
)
from .circuit import ProgramCircuit

warnings.filterwarnings("ignore", category=DeprecationWarning)

__all__ = ["CotProg", "CoTProgParser"]

def safe_getsource(f):
    """Get the underlying source code.

    Arguments:
        f: The program and code to parse.

    """
    try:
        import inspect

        return inspect.getsource(f)
    except OSError:
        try:
            import dill.source as ds

            return ds.getsource(f)
        except Exception:
            raise RuntimeError(
                "Cannot retrieve source code. "
                "Is this code being run interactively (REPL/Jupyter) or after reload? "
                "Move your function to a script file for @CoTProg."
            )

@dataclass
class CotProg:
    """A representation of the CoT program.

    Attributes:
        program: A string rendering of the underlying program,
        native_python: A native python translation of the program

    Methods:
        execute(self): Executes the native python program.
        sample(self, tries: int): Samples `tries` number of random programs.
        semantics(self, pretty): Computes a sympy semantucs for program.

    """
    _prog: Callable
    _cot_prog: str
    _py_prog: str
    _prop_vals: Dict[str, str]
    _variables: Set[str]
    _decision_vars: Dict[str, Tuple[float, str]]
    _return_vars: Dict[str, str]

    ## semantic attributes 
    _semantics: Optional[Any] = None
    _evidence: Optional[Any] = None
    _circuit: Optional[Any] = None

    ## weights 
    _weights: Optional[torch.tensor] = None
    _weight_funcs: Optional[Dict[str, Tuple[Callable, List[str]]]] = None
    _categoricals: Optional[Set[str]] = None

    @property
    def weights(self):
        return self._weights
    @property 
    def sec_weights(self):
        return self._sec_weights

    @property
    def prog(self):
        return self._prog
    
    def __post_init__(self):
        if self._weights is None: 
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._weights = torch.tensor(
                [i[0] for vname, i in self._decision_vars.items()], device=device,
            )

    def __call__(self):
        return self.execute()

    def execute(self) -> str:
        """Executes the underlying native Python program. Note: 
        this can only run certain types of probabilistic programs, 
        specifically ones without `observe` statements. 
        """
        return self._prog()

    def sample(self, tries: int = 3000) -> Dict[str, float]:
        """Runs the native program `tries` number of times and records results,
        including number of exceptions caught, counts of specific return values.


        Arguments:
            tries: The number of times to run the code.
        Returns:
            A dictionary containing the counts and other details of the run.

        """
        attempts = 0.0
        executed = 0.0
        return_values = defaultdict(int)

        for _ in range(tries):
            attempts += 1.0
            try:
                r = self.execute()
                executed += 1.0
                return_values[r] += 1.0
            except (ValueError, AssertionError):
                continue

        metrics = {
            "attempts": attempts,
            "return_values": dict(return_values),
            "executed": executed,
        }
        return metrics

    @property
    def native_python(self) -> str:
        """Returns a string version of the native python code underlying the program."""
        return self._py_prog

    @property
    def program(self) -> Callable:
        """Returns the executable Python program."""
        return self._prog

    def _build_semantics(self, stype='sympy', weight_id: Optional[int] = None):
        """Method for building different semantic representations of programs.

        Arguments: 
            stype: The particular semantic object to build.
        Raises: 
            ValueErrorr when unknown semantic type is encountered. 
        """
        tree = ast.parse(self._py_prog)
        if weight_id is not None: 
            pass 

        sem = ProgramToSemantics(
            mapper=stype,
            string_assignments=self._prop_vals,
            decision_assignments=self._decision_vars,
            return_vars=list(self._return_vars.values()),
            var_list=self._variables,
        )
        sem.visit(tree)
        return sem.compute_semantics()    
    
    def semantics(self, pretty=False) -> Union[sympy.Symbol, str]:
        """Return program semantics, optionally in pretty print form

        Arguments:
            pretty: Return a pretty printed string version.
        Returns:
            Either a string representation of the sympy formula if
                `pretty` is set to true, otherwise just the sympy formula.
        """
        if self._semantics is None:
            self._semantics = self._build_semantics(stype='sympy')
            
        if pretty:
            clauses = ""
            for item in self._semantics.args:
                clauses += f"\n\t{item},"
            return f"And({clauses}\n)"

        return self._semantics

    def compile_circuit(self, weight_id: Optional[int] = None) -> ProgramCircuit:
        """Compiles the program semantics into a tractable SDD
        circuit.

        """
        if self._circuit is None or weight_id is not None:
            self._circuit = self._build_semantics(stype='sdd', weight_id=weight_id)
        return self._circuit

    def inference(
        self, 
        mode: Optional[str] = "prob", 
        query: Optional[set] = set(), 
        compute_grad: Optional[bool] = False,
        weight_id: Optional[int] = None,
    ):
        """Computes inferences on the program via circuits. 

        Arguments: 
            mode: The particular mode of reasoning to perform (e.g., probabilistic, 
                fuzzy reasoning, gradient).
            query: An (optional) propositional query to the circuit. 
            compute_grad: Whether to compute gradients during inference.
        """
        circuit = self.compile_circuit(weight_id=weight_id)
        return circuit.inference(
            mode=mode, 
            marginals=query, 
            has_evidence=bool(self._evidence), 
            compute_grad=compute_grad, 
            numeric_funcs=self._weight_funcs,
            categoricals=self._categoricals,
            weight_id=weight_id,
        )

class CoTProgParser:
    """Decorator class for parsing CoTProg programs and extracting normalized
    Python programs and semantics from those programs.

    Methods:
        parse(f, name): Main method for parsing input programs (either provided
            as strings or `Callable` objects.

    """

    def __init__(self):
        self.transforms = [CoTProgramTransformer]

    def parse(self, f: Union[str, Callable], name: Optional[str] = "cot") -> CotProg:
        """Parses a CoTProgram.

        Arguments:
            f: The target program to parse.
            name: The (optional) name of the program.

        Raises: 
            ValueError when program cannot be parsed by Python parser. 

        """
        glbs = {}
        glbs.update(PROBABILITY_VARIABLES)
        glbs.update(PROBABILITY_FUNCTIONS)
            
        if not isinstance(f, str):
            orig_source = safe_getsource(f)
            func_name = f.__name__
            
        else:
            orig_source = f
            func_name = name

        lines = textwrap.dedent(orig_source).lstrip().splitlines()
        lines = [line for line in lines if not line.strip().startswith("@CoTProg")]
        source = "\n".join(lines)

        ## source code transformations
        try: 
            tree = ast.parse(source)
        except Exception as e:
            raise ValueError(f'Program not parseable by Python parser: {e}')
            
        for tclass in self.transforms:
            transformer = tclass()
            tree = transformer.visit(tree)
            ast.fix_missing_locations(tree)

        ## collect variables, assignments, etc. 
        meta = VarCollector({}, {}, {}, set())
        meta.visit(tree)
        num_funcs = meta.numeric_funcs

        ## inline probability references like `P1.prob` based on collected decisions
        prob_inliner = ProbRefRewriter(meta.decision_assignments)
        tree = prob_inliner.visit(tree)
        ast.fix_missing_locations(tree)

        py_prog = astor.to_source(tree)

        #print(py_prog)
        code = compile(tree, filename="<ast>", mode="exec")

        def wrapper(*args, **kwargs) -> Callable:
            locs = {}
            exec(code, glbs, locs)
            return locs[func_name](*args, **kwargs)
        
        return CotProg(
            _prog=wrapper,
            _cot_prog=orig_source,
            _py_prog=py_prog,
            _prop_vals=meta.string_vars,
            _variables=list(set(meta.variables)),
            _decision_vars=meta.decision_assignments,
            _return_vars=meta.return_vals,
            _weight_funcs=num_funcs,
            _categoricals=meta.categoricals,
        )
    
    def __call__(
        self, f: Union[str, Callable], name: Optional[str] = "cot"
    ) -> CotProg:
        return self.parse(f, name)