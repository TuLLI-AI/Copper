from abc import ABC
import sympy
import torch
from sympy import *
from pysdd.sdd import SddManager, SddNode
from typing import Union, Callable, Dict, List, Tuple, Optional, Set, Any, Type
from .circuit import ProgramCircuit

__all__ = [
    "construct_formula",
    "SympyMapper",
    "SymbolicMapper",
]

class SymbolicMapper(ABC):
    """Base class for symbolic mappings (i.e., clases that build symbolic 
    objects such as formulas, circuits, ...)

    Attributes: 
        num_variables: The number of variables in the program and logic encoding.
        manager: Optional manager attribute (for Sdd mappers). 
        false: Representation of logical `false` for the mapper. 
        true: Representation of logical `true` for this mapping.

    Methods: 
        Or: implementation of disjunction. 
        And: Implementation of conjunction 
        Symbol: Maps string input to a symbol in the underlying system. 
        Not: Implementation of negation. 

    """
    
    def __init__(
        self,
        var_list: Optional[List[str]] = [],
        decision_vars: Optional[List[str]] = [],
        return_vars: Optional[List[str]] = [],
    ) -> None:
        """Initializes the mapper instance. 

        Arguments: 
            num_variables: The number of variables in the program and 
                compilation. 
            manager: Optional manager for mapper (e.g., Sdd circuit mananger). 
        """
        self.var_list = var_list
        self.decision_vars = decision_vars
        self.return_vars = return_vars
        self.sym_map = {}
        
    def __getitem__(self, s: str):
        ### returns underlying symbolic encoding for a symbol
        if s not in self.sym_map:
            raise ValueError(f"Unknown symbol: {s}")
        return self.sym_map[s]

    def Or(self, obj1, obj2):
        """Implementation of logical `Or`.

        Arguments: 
            obj1: The first argument for disjunction. 
            obj2: The second argument for disjunction
        """
        raise NotImplementedError
    
    def And(self, obj1, obj2):
        """Implementation of logical `And`.

        Arguments: 
            obj1: The first argument for conjunction. 
            obj2: The second argument for conjunction. 
        """
        raise NotImplementedError 

    def Implies(self, obj1, obj2):
        """Implementation of logical material implication. 

        Arguments: 
            obj1: The first argument for implication. 
            obj2: The second argument for implication. 
        """
        raise NotImplementedError 

    def Biconditional(self, obj1, obj2):
        """Implementation of the biconditional.

        Arguments: 
            obj1: The first argument for implication. 
            obj2: The second argument for implication. 
        """
        raise NotImplementedError

    def Not(self, expr):
        """Implementation of logical negation.

        Arguments: 
            expr: The expression to negate. 
        """
        raise NotImplementedError

    def Symbol(self, name: str):
        """Representation and creation on primitive symbols.

        Arguments: 
            name: The name of the symbol. 
        """
        raise NotImplementedError
    
    @property
    def false(self):
        """Internal representation of the false primitive."""
        raise NotImplementedError

    @property
    def true(self):
        """Internal representation of the true primitive."""
        raise NotImplementedError

    def _post_process(self, formula):
        return formula 

    def construct_formula(
        self, guards: Dict[Any, Any], decisions, evidence: Optional[List[Any]] = []
    ):
        """Construct a symbolic formula or object based on guards.
        
        Arguments: 
            guards: The input guards. 
            decisions: The decision variables. 
            evidence: The background evidence. 
        """
        final_formula = self.true

        for var, rules in guards.items():
            if len(rules) > 1 and set(rules) == {self.true}:
                pass
            if rules == [self.true] and var in decisions:
                continue

            g = self.false
            for r in rules:
                g = self.Or(g, r)

            var_obj = self[var] 
            if var not in decisions:
                ## could replace with `biconditional` 
                full_guard = self.And(
                    self.Implies(var_obj, g), self.Implies(g, var_obj)
                )
                final_formula = self.And(final_formula, full_guard)

        if evidence:
            final_evidence = self.true
            for evd in evidence:
                final_evidence = self.And(final_evidence, evd)
            final_formula = self.And(final_formula, final_evidence)

        return self._post_process(final_formula) 

class SympyMapper(SymbolicMapper):
    """Symbolic mapper object that maps into sympy. 

    Attributes: 
        num_variables: The number of variables in the program and logic encoding.
        manager: Optional manager attribute (for Sdd mappers)
        false: Representation of logical `false` for the mapper. 
        true: Representation of logical `true` for this mapping.
    
    """
    def Or(self, obj1: sympy.Symbol, obj2: sympy.Symbol) -> sympy.Or:
        """Implementation of logical `Or`.

        Arguments: 
            obj1: The first argument for disjunction. 
            obj2: The second argument for disjunction
        """
        return sympy.Or(obj1, obj2)
    
    def And(self, obj1: sympy.Symbol, obj2: sympy.Symbol) -> sympy.And:
        """Implementation of logical `And`.

        Arguments: 
            obj1: The first argument for conjunction. 
            obj2: The second argument for conjunction. 
        """
        return sympy.And(obj1, obj2)
    
    def Implies(self, obj1: sympy.Symbol, obj2: sympy.Symbol) -> sympy.Implies:
        """Implementation of logical material implication. 

        Arguments: 
            obj1: The first argument for implication. 
            obj2: The second argument for implication. 
        """
        return sympy.Implies(obj1, obj2)
    
    def Symbol(
        self, name: str, return_val: Optional[bool] = False
    ) -> Union[str, sympy.Symbol]:
        """Representation and creation on primitive symbols. Returns either 
        internal string symbol or the underlying object (if `return_val=True`).  

        Arguments: 
            name: The name of the symbol. 
            return_val: Switch to determine whether to return symbol object. 
        """
        self.sym_map[name] = sympy.Symbol(name)
        if return_val:
            return self.sym_map[name]
        return name
    
    def Not(self, expr: sympy.Symbol) -> sympy.Not:
        """Implementation of logical negation.

        Arguments: 
            expr: The expression to negate. 
        """
        return sympy.Not(expr)
    
    @property
    def false(self) -> sympy.false:
        """Internal representation of the false primitive."""
        return sympy.false
    
    @property
    def true(self) -> sympy.true:
        """Internal representation of the true primitive."""
        return sympy.true

class SddMapper(SymbolicMapper):
    """Symbolic mapper object that maps into sdd circuits. 

    Attributes: 
        num_variables: The number of variables in the program and logic 
            encoding.
        manager: Optional manager attribute (for Sdd mappers)
        false: Representation of logical `false` for the mapper. 
        true: Representation of logical `true` for this mapping.
    """
    def __init__(self,**kwargs):
        """Initializes the mapper instance. 

        Arguments: 
            num_variables: The number of variables in the program and 
                compilation. 
            manager: Optional manager for mapper (e.g., Sdd circuit mananger). 
        """
        super().__init__(**kwargs)
        self._init_manager()

    def _init_manager(self) -> None:
        """Initializes circuit object, manager and input variables. Note that SDD 
        circuits requires knowing the number of variables at the start. 
        """
        ## Build an explicit (name, kind, prob) entry per SDD variable.
        ## `kind` travels with the name so the probability a literal receives
        ## is never decided by a name lookup: two variables that happen to
        ## share a name can no longer steal each other's weight.
        ##
        ## The aux suffix uses a `__` separator because a bare f"{var}{k+2}"
        ## made the second slot of Flip "P2" collide with a primary step
        ## literally named "P22" (and "P1"'s with "P12"). On a collision the
        ## name lookup below hit `decision_vars` first, so the aux literal was
        ## bound to the *primary's* probability and the primary was shadowed
        ## in `sym_map`; `lit_map` also ended up with two literals sharing a
        ## name, which duplicated keys in the reported gradients.
        entries = []
        for var in self.var_list:
            details = self.decision_vars.get(var, None)
            if details is not None:
                entries.append((var, "primary", details[0]))
            else:
                entries.append((var, "bool", None))
        ## aux weight variables if any exist — appended after the primaries so
        ## the literal ordering is unchanged.
        for var, (_, _, _, extra_probs) in self.decision_vars.items():
            for k, prob in enumerate(extra_probs):
                entries.append((f"{var}__{k+2}", "aux", prob))

        names = [e[0] for e in entries]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"duplicate SDD variable name(s) {dupes}: every program "
                f"variable and auxiliary weight slot must have a unique name"
            )

        self.var_list = names
        num_vars = len(self.var_list)
        self.manager = SddManager(var_count=num_vars)
        self.var_lookup = {}
        self.lit_map    = {}
        self.prob_list   = []
        self.sec_prob_list = []
        self.bool_set    = set()
        self.weight_map  = {}

        for v, (var, kind, prob) in enumerate(entries):
            identifier = self.manager[v + 1]
            self.sym_map[var] = identifier
            self.lit_map[-identifier.literal] = f"~{var}"
            self.lit_map[identifier.literal] = var

            if kind == "bool":
                self.bool_set.add(identifier.literal)
            else:
                self.prob_list.append(prob)
                self.weight_map[identifier.literal] = len(self.prob_list) - 1


    def _post_process(self, formula):
        return ProgramCircuit(
            mgr=self.manager,
            circuit=formula,
            literal_map=self.lit_map,
            weight_map=self.weight_map,
            bool_set=self.bool_set,
            weights=torch.tensor(self.prob_list),
            return_vars=self.return_vars,
        )
    
            
    def Or(self, obj1: SddNode, obj2: SddNode) -> SddNode:
        """Implementation of logical `Or`.

        Arguments: 
            obj1: The first argument for disjunction. 
            obj2: The second argument for disjunction
        """
        return self.manager.disjoin(obj1, obj2)
    
    def And(self, obj1: SddNode, obj2: SddNode) -> SddNode:
        """Implementation of logical `And`.

        Arguments: 
            obj1: The first argument for conjunction. 
            obj2: The second argument for conjunction. 
        """
        return self.manager.conjoin(obj1, obj2)

    def Implies(self, obj1: SddNode, obj2: SddNode) -> SddNode:
        """Implementation of logical material implication. 

        Arguments: 
            obj1: The first argument for implication. 
            obj2: The second argument for implication. 
        """
        return self.manager.disjoin(~obj1, obj2)
    
    def Symbol(
        self, name: str, return_val: Optional[bool] = False
    ) -> Union[str, SddNode]:
        """Representation and creation on primitive symbols.

        Arguments: 
            name: The name of the symbol. 
        """
        
        intn = self.sym_map.get(name,None)
        if intn is None:
            raise ValueError(f'Unknown circuit variable encountered: {name}')
        if return_val:
            return self.sym_map[name]
        return name
 
    def Not(self, expr: SddNode) -> SddNode:
        """Implementation of logical negation.

        Arguments: 
            expr: The expression to negate. 
        """
        return ~expr
    
    @property
    def false(self) -> SddNode:
        """Internal representation of the false primitive."""
        return self.manager.false()
    
    @property
    def true(self) -> SddNode:
        """Internal representation of the true primitive."""
        return self.manager.true()


class Z3Mapper(SymbolicMapper):
    """Mappings for translation of Boolean formulas into the Z3 language"""


class PySatMapper(SymbolicMapper):
    """Mappings for translation of Boolean formulas into the PySat language"""
    


# class MapperRegistry(object):
#     IMPLEMENTATIONS: 
#     def __call__(self,*args, **kwargs):
#         class_to_add = args[0]

MAPPERS = {
    "sympy" : SympyMapper,
    "sdd"   : SddMapper,
}

def BuildMapper(mapper_name: str, **kwargs) -> Type[SymbolicMapper]:
    """Factory for building mapper instance

    Arguments: 
        mapper_name: The name of the mapper. 

    Raises: 
        ValueError when unknown mapper name is given. 
    """
    mcls = MAPPERS.get(mapper_name, None)
    if mcls is None:
        raise ValueError(f'Uknown mapper type: {mapper_name}')

    return mcls(**kwargs) 
