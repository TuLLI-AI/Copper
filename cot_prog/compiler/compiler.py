import numpy as np
import ast
import torch
import math
import re
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Union, Dict, List, Tuple, Optional, Set, Any
from .sem import BuildMapper

__all__ = [
    "ProgramToSemantics",
    "CoTProgramTransformer",
    "VarCollector",
    "ProbRefRewriter",
    "PROBABILITY_FUNCTIONS",
    "PROBABILITY_VARIABLES",
]

def Decision(prop: str, prob: float, prob2: float = None) -> bool:
    """Make a random decision based on provided probability.

    Arguments:
        prop: The underlying proposition involved in the decision (ignored)
        prob: The probability assign with that prop being true/false based on
            on LLM.
        prob2: An optional second probability for more complex decision-making.
    Returns:
        A True/False boolean based on the provided probability.
    """
    return np.random.choice([True, False], p=[prob, 1 - prob])

def CategoricalDecision(prop: str, args) -> list[bool]:
    """Make a categorical random decision based on provided probabilities.

    Arguments:
        prop: The underlying proposition involved in the decision (ignored)
        args: A list of probabilities for each category, which should sum to 1.
    Returns:
        A list of boolean values indicating the selected category.
    """
    selected = np.random.choice(len(args), p=args)
    return [True if str(i) == str(selected) else False for i in range(len(args))]

def Bernoulli(prop: str, prob: float, prob2: float = None) -> bool:
    """Alias for `Decision`."""
    return Decision(prop, prob, prob2)

def Flip(prop: str, prob: float, prob2: float = None) -> bool:
    """Alias for `Decision`."""
    return Decision(prop, prob, prob2)

def Factor(prop: str, prob: float, prob2: float = None) -> bool:
    """Alias for `Decision`."""
    if prob2 is not None: 
        raise ValueError("Factor does not support two probabilities.")
    return Decision(prop, prob, prob2)

def Categorical(prop: str, *args):
    """Categorical distribution for multinomial decisions. 

    Arguments:
        prop: The underlying proposition involved in the decision (ignored)
        prob: A list of probabilities for each category, which should sum to 1.
    Returns:    
    An integer index corresponding to the sampled category.
    """
    if len(args) == 1 and not isinstance(args[0], list):
        return Decision(prop, args[0])
    return CategoricalDecision(prop, *args)

def evidence(bool_expr):
    """Evidence predicate. Does nothing in underlying Python. 

    Arguments: 
        bool_expr: The boolean formula or expression that encodes evidence or 
            background constraints. 
    """
    pass

def observe(bool_expr):
    """object predicate and operator. Does nothing in underlying Python (alias of 
    the evidence operator defined above). 

    Arguments: 
        bool_expr: The boolean formula or expression that encodes evidence or 
            background constraints.
    """
    pass

KEYWORDS = {
    "ValueError" : "Exc",
}
BERNOULLI_NAMES = set([
    "Decision",
    "Bernoulli",
    "Flip",
    "Factor",
    "Categorical",
])
PROBABILITY_VARIABLES = {
    "Decision"    : Decision, 
    "evidence"    : evidence,
    "observe"     : observe,
    "Flip"        : Flip,
    "Bernoulli"   : Bernoulli,
    "Factor"      : Factor,
    "Categorical" : Categorical,
}
def _dep_var_name(ref: str) -> str:
    """Map a probability reference to its SDD variable name.

    ``X.p`` / ``X.prob``      -> ``X``            (the primary slot)
    ``X.p2`` / ``X.prob2``    -> ``X__2``         (an auxiliary slot)

    The auxiliary spelling MUST match sem.py::_init_manager, which registers
    those variables as f"{var}__{k+2}". This used to be done with
    ``ref.replace(".prob","").replace(".p","")``, which produced ``X2`` and so
    silently aliased ``P2.p2`` onto a different step literally named ``P22``.
    """
    m = re.fullmatch(r'(.+)\.p(?:rob)?(\d+)', ref)
    if m:
        return f"{m.group(1)}__{m.group(2)}"
    m = re.fullmatch(r'(.+)\.p(?:rob)?', ref)
    if m:
        return m.group(1)
    return ref


EPS = 1e-7

def _logit(x: float) -> float:
    x = max(EPS, min(1 - EPS, x))
    return math.log(x / (1 - x))


def LogitPool(*xs: float | torch.Tensor) -> float | torch.Tensor:
    """Predefined probability function: logistic(sigmoid) of sum of inputs.

    Example:
        LogitPool(0.8) -> sigmoid(0.8)
        LogitPool(0.8, 0.3) -> sigmoid(1.1)
    """
    ## check if torch input and perform torch-based computation if so
    if xs and any(isinstance(x, torch.Tensor) for x in xs):
        # Ensure all inputs are tensors on the graph
        tensors = [x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32) for x in xs]
        stacked = torch.stack(tensors)
        stacked = torch.clamp(stacked, min=1e-7, max=1 - 1e-7)
        logits = torch.log(stacked / (1 - stacked))  # inverse sigmoid = logit
        return torch.sigmoid(torch.sum(logits) / len(xs))
    
    sigmoid = lambda x: 1.0 / (1.0 + math.exp(-x))
    
    return sigmoid(sum(_logit(x) for x in xs) / len(xs)) if xs else 0.0

def NoisyOr(*xs: float | torch.Tensor) -> float | torch.Tensor:
    """Predefined probability function: Noisy-OR of inputs.

    Example:
        NoisyOr(0.8, 0.3) -> 1 - (1-0.8)(1-0.3) = 0.84
    """
    if xs and any(isinstance(x, torch.Tensor) for x in xs):
        ### ensure all inputs are tensors on the graph
        tensors = [x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32) for x in xs]
        stacked = torch.stack(tensors)
        return 1 - torch.prod(1 - stacked) if xs else torch.tensor(0.0)
        
    return 1 - np.prod([1 - x for x in xs]) if xs else 0.0

def Combine(*xs: float | torch.Tensor) -> float | torch.Tensor:
    """Predefined probability function: simple average of inputs."""
    if len(xs) > 2:
        raise ValueError("Combine function only supports up to 2 inputs for now.")
    
    if xs and any(isinstance(x, torch.Tensor) for x in xs):
        ## ensure all inputs are tensors on the graph
        tensors = [x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32) for x in xs]
        stacked = torch.stack(tensors)
        return xs[0] / torch.sum(stacked)
    
    return xs[0] / sum(xs) if xs else 0.0

def Mixture(*xs: float | torch.Tensor) -> list[float] | torch.Tensor:
    """Combine weights for form categorical distribution with more than 2 categories. Simple assigns 
    each probability `p` as `p / Z` where `Z` is the sum of all probabilities. 

    """    
    if xs and any(isinstance(x, torch.Tensor) for x in xs):
        tensors = [x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32) for x in xs]
        stacked = torch.stack(tensors)
        return stacked / torch.sum(stacked) if xs else torch.tensor(0.0)
    return [x / sum(xs) for x in xs] if xs else [0.0]

def Mean(*xs: float | torch.Tensor) -> float | torch.Tensor:
    """Predefined probability function: simple average of inputs."""
    if xs and any(isinstance(x, torch.Tensor) for x in xs):
        ### ensure all inputs are tensors on the graph
        tensors = [x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32) for x in xs]  
        stacked = torch.stack(tensors)
        return torch.mean(stacked) if xs else torch.tensor(0.0)
    
    return np.mean(xs) if xs else 0.0 

def Max(*xs: float | torch.Tensor) -> float | torch.Tensor:
    """Predefined probability function: max of inputs."""
    if xs and any(isinstance(x, torch.Tensor) for x in xs):
        tensors = [x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32) for x in xs]
        stacked = torch.stack(tensors)
        return torch.max(stacked) if xs else torch.tensor(0.0)
    
    return max(xs) if xs else 0.0

def Min(*xs: float | torch.Tensor) -> float | torch.Tensor:
    """Predefined probability function: min of inputs."""
    if xs and any(isinstance(x, torch.Tensor) for x in xs):
        tensors = [x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32) for x in xs]
        stacked = torch.stack(tensors)
        return torch.min(stacked) if xs else torch.tensor(0.0)
    
    return min(xs) if xs else 0.0  

def Prod(*xs: float | torch.Tensor) -> float | torch.Tensor:
    """Predefined probability function: product of inputs."""
    if xs and any(isinstance(x, torch.Tensor) for x in xs):
        tensors = [x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32) for x in xs]
        stacked = torch.stack(tensors)
        return torch.prod(stacked) if xs else torch.tensor(1.0)
    
    return np.prod(xs) if xs else 1.0

def Sum(*xs: float | torch.Tensor) -> float | torch.Tensor:
    """Predefined probability function: sum of inputs."""
    if xs and any(isinstance(x, torch.Tensor) for x in xs):
        tensors = [x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32) for x in xs]
        stacked = torch.stack(tensors)
        return torch.sum(stacked) if xs else torch.tensor(0.0)
    
    return np.sum(xs) if xs else 0.0

def DisCalibratedPrior(*xs: float | torch.Tensor) -> float | torch.Tensor:
    """Calibrate prior using posterior disagreement only when posterior is less confident.

    Assumes:
        xs[0] = prior p(a)
        xs[1] = direct posterior p(a | e)

    Shrinks the prior toward 0.5 only when the direct posterior is closer
    to 0.5 than the prior is.

    This avoids shrinking a low prior upward when the posterior is even lower.
    """
    if len(xs) != 2:
        raise ValueError(
            "ConfidenceDisagreementCalibratedPrior requires exactly two inputs: "
            "prior, direct_posterior"
        )

    prior, direct_post = xs

    if any(isinstance(x, torch.Tensor) for x in xs):
        t_prior = prior if isinstance(prior, torch.Tensor) else torch.tensor(prior, dtype=torch.float32)
        t_post = direct_post if isinstance(direct_post, torch.Tensor) else torch.tensor(direct_post, dtype=torch.float32)

        prior_logit = torch.logit(t_prior, eps=1e-6)
        post_logit = torch.logit(t_post, eps=1e-6)

        prior_conf = torch.abs(prior_logit)
        post_conf = torch.abs(post_logit)

        # Positive only when posterior is less confident than the prior.
        excess_prior_conf = torch.relu(prior_conf - post_conf)

        # Hyperparameter-free shrinkage gate.
        trust_prior = torch.exp(-excess_prior_conf)

        calibrated_logit = trust_prior * prior_logit
        return torch.sigmoid(calibrated_logit)

    eps = 1e-6
    p = min(max(float(prior), eps), 1.0 - eps)
    r = min(max(float(direct_post), eps), 1.0 - eps)

    prior_logit = math.log(p / (1.0 - p))
    post_logit = math.log(r / (1.0 - r))

    prior_conf = abs(prior_logit)
    post_conf = abs(post_logit)

    excess_prior_conf = max(0.0, prior_conf - post_conf)

    trust_prior = math.exp(-excess_prior_conf)

    calibrated_logit = trust_prior * prior_logit
    return 1.0 / (1.0 + math.exp(-calibrated_logit))

def DisCalibration(*xs: float | torch.Tensor) -> float | torch.Tensor:
    """Calibrate a prior using disagreement with a direct posterior estimate.

    Assumes:
        xs[0] = prior p(a)
        xs[1] = direct posterior p(a | e)

    Idea:
        prior_logit = logit p(a)
        post_logit  = logit p(a | e)

        disagreement = |prior_logit - post_logit|
        trust_prior  = exp(-disagreement)

        calibrated_prior_logit = trust_prior * prior_logit
        calibrated_prior = sigmoid(calibrated_prior_logit)

    Behavior:
        - If prior and posterior agree, trust_prior ≈ 1, so output ≈ prior.
        - If they disagree strongly, trust_prior ≈ 0, so output ≈ 0.5.
        - The posterior is used as a calibration/sensitivity signal, not as a replacement prior.
    """
    if len(xs) != 2:
        raise ValueError("DisagreementCalibratedPrior requires exactly two inputs: prior, direct_posterior")

    prior, direct_post = xs

    if any(isinstance(x, torch.Tensor) for x in xs):
        t_prior = prior if isinstance(prior, torch.Tensor) else torch.tensor(prior, dtype=torch.float32)
        t_post = direct_post if isinstance(direct_post, torch.Tensor) else torch.tensor(direct_post, dtype=torch.float32)

        # torch.logit with eps avoids infinities at 0 or 1.
        prior_logit = torch.logit(t_prior, eps=1e-6)
        post_logit = torch.logit(t_post, eps=1e-6)

        disagreement = torch.abs(prior_logit - post_logit)

        # Hyperparameter-free trust gate.
        trust_prior = torch.exp(-disagreement)

        calibrated_logit = trust_prior * prior_logit
        return torch.sigmoid(calibrated_logit)

    # Float path.
    eps = 1e-6
    p = min(max(float(prior), eps), 1.0 - eps)
    r = min(max(float(direct_post), eps), 1.0 - eps)

    prior_logit = math.log(p / (1.0 - p))
    post_logit = math.log(r / (1.0 - r))

    disagreement = abs(prior_logit - post_logit)
    trust_prior = math.exp(-disagreement)

    calibrated_logit = trust_prior * prior_logit
    return 1.0 / (1.0 + math.exp(-calibrated_logit))

# def Softmax(*xs: float | torch.Tensor) -> float | torch.Tensor:
#     """Predefined probability function: softmax of inputs."""
#     if xs and any(isinstance(x, torch.Tensor) for x in xs):
#         return F.softmax(torch.tensor(xs), dim=0)
#     exps = np.exp(xs)
#     return exps / np.sum(exps) if xs else 0.0

PROBABILITY_FUNCTIONS = {
    "LogitPool"        : LogitPool,
    "NoisyOr"          : NoisyOr,
    "Combine"          : Combine,
    "Mean"             : Mean,
    "Max"              : Max,
    "Min"              : Min,
    "Prod"             : Prod,
    "Sum"              : Sum,
    "Mixture"          : Mixture,
    "Calibrate"        : DisCalibration,
    "DCalibrate"       : DisCalibratedPrior, 
    #"Softmax"  : Softmax,
}
ATTR_FUNCTIONS = {
    "mean"  : Mean,
    "m"     : Mean,
    "pool"  : LogitPool,
    "max"   : Max,
    "min"   : Min,
    "prod"  : Prod,
    "cal"   : DisCalibration,
    "dcal"  : DisCalibratedPrior, 
}

def extract_cond(
    mapper,
    *,
    test: Union[ast.BoolOp, ast.Name, ast.UnaryOp, ast.Constant],
    bools,
    relax: Optional[bool] = False,
    ):
    """Extracts the `if` conditions in an if-then construction (ast) and translates to mapper object.

    Arguments:
        mapper: The particular symbolic mapper being used. 
        test: The node containing the if test code.
        bools: The program boolean variables. 
        relax: Parse formuals/conditions without checking if all variables 
            are defined (used for parsing global `observe` statements). 
    Raises:
        SyntaxError when encodig an unsopported test case.
        ValueError when condition is not a Bool or an unknown variable.

    """
    if isinstance(test, ast.BoolOp):
        cond_formulas = [
            extract_cond(mapper, test=i, bools=bools, relax=relax) for i in test.values
        ]
        if isinstance(test.op, ast.And):
            op = mapper.And
        elif isinstance(test.op, ast.Or):
            op = mapper.Or
        else:
            raise ValueError(f"Unknown Boolean op: {test.op}")
            
        formula = None
        for f in cond_formulas:
            formula = f if formula is None else op(formula, f)
        return formula
    
    elif isinstance(test, ast.Name):
        if (not relax) and (test.id not in bools):
            raise ValueError(
                f"Unknown variable used in bool condition: {test.id}"
            )
        return mapper.Symbol(KEYWORDS.get(test.id, test.id), return_val=True)
    
    elif isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return mapper.Not(extract_cond(mapper, test=test.operand, bools=bools, relax=relax))

    ## primitive `True` and `False`
    elif isinstance(test, ast.Constant) and (test.value is True):
        return mapper.true
    
    elif isinstance(test, ast.Constant) and (test.value is False):
        return mapper.false
    
    else:
        raise SyntaxError(f"Unsupported if-test: {ast.dump(test)}")
    
class VarCollector(ast.NodeVisitor):
    """Visit and collect variable values. Assumes a fully typed Boolean
    language as constructed via the `CoTProgramTransformer` below.

    Attributes:
        string_vars: A collection of the string variables in program.
        decision_assignments: Associated Boolean (or random variables) with
            strings and probabilities for bookeeping purposes.
    """

    def __init__(
        self, string_vars = None, decision_assignments = None, return_vals = None, exception_vars = None
    ) -> None:
        """Creates VarCollector instance for collecting variables (strings, bools)
        and also checking that all variables are assignment.

        Arguments:
            string_vars: A collection of the string variables in program.
            decision_assignments: Associated Boolean (or random variables) with
                strings and probabilities for bookeeping purposes.
            return_vals: The provided return values. 
            exception_vars: Records if any exceptions are used in the program. 
        """
        self.string_vars = string_vars if string_vars else {}
        self.decision_assignments = decision_assignments if decision_assignments else {}
        self.return_vals = return_vals if return_vals else {}
        self.exception_vars = exception_vars if exception_vars else set()
        self.categoricals = set()
        self.numeric_funcs = {}      

    @property
    def variables(self) -> List[str]:
        """Returns the full set of variables in the program."""
        return list(self.exception_vars) + list(set(self.return_vals.values())) + list(self.decision_assignments.keys())

    def visit_Raise(self, node):
        self.exception_vars.add("Exc")
    def visit_Assert(self, node):
        self.exception_vars.add("Exc")

    def visit_Assign(self, node: ast.Assign):
        """Collects variables from non-annotated assignments. Only supports categorical variables with tuple assignment, 
        e.g., `X, Y = Categorical(...)` for now.
        """ 
        if not (node.value.func.id == "Categorical" and len(node.value.args) >= 2):
            raise ValueError(f"Only tuple assignment from Categorical distribution supported, invalid line: {ast.unparse(node)}")

        vars = []
        for target in node.targets:
            vars.extend(self.extract_names(target))

        for index,var in enumerate(vars):
            prob = self.eval_prob_expr(node.value.args[1], var, index)
            if var in self.decision_assignments:
                raise ValueError(
                    f"Decision variable defined multiple times: {var} in line: {ast.unparse(node)}. Please change name"
                )
            str_val = f"{self.get_value(node.value.args[0])}_{var}"
            self.decision_assignments[var] = (prob, None, str_val, [])
            self.categoricals.add(var)

    def extract_names(self, target) -> list[str]:
        if isinstance(target, ast.Name):
            return [target.id]
        elif isinstance(target, ast.Tuple):
            names = []
            for elt in target.elts:
                names.extend(self.extract_names(elt))
            return names
        elif isinstance(target, ast.Starred):
            return self.extract_names(target.value)
        else:
            raise ValueError(f"Unsupported assignment target: {ast.dump(target)}")

    def get_value(self, arg) -> str:
        """Gets the value of a function argument.

        Arguments:
            arg: The incoming ast node.
        """
        if isinstance(arg, ast.Constant):
            return arg.value
        elif isinstance(arg, ast.Name):
            return arg.id
        else:
            return ast.unparse(arg)

    def eval_prob_expr(self, expr: ast.AST, var_name: str = None, index: int = -1) -> float:
        """Safely evaluates numeric probability expressions.

        Supports:
            - Numeric literals (int/float)
            - Unary minus on numeric values
            - Binary ops: +, -, *, /
            - Calls to whitelisted functions in `PROBABILITY_FUNCTIONS`
            - Attribute references of form `<DecisionVar>.prob`
        Raises:            
            ValueError for unknown variables/functions or unsupported syntax.
        
        """
        # Numeric literal
        if isinstance(expr, ast.Constant) and isinstance(expr.value, (int, float)):
            return float(expr.value)

        # Unary minus
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
            return -self.eval_prob_expr(expr.operand)

        # Binary arithmetic
        if isinstance(expr, ast.BinOp):
            left = self.eval_prob_expr(expr.left)
            right = self.eval_prob_expr(expr.right)
            if isinstance(expr.op, ast.Add):
                return left + right
            if isinstance(expr.op, ast.Sub):
                return left - right
            if isinstance(expr.op, ast.Mult):
                return left * right
            if isinstance(expr.op, ast.Div):
                return left / right
            raise ValueError(f"Unsupported operator in probability expression: {ast.dump(expr.op)}")
        
        ## list of probability expressions 
        if isinstance(expr, ast.List):
            return [self.eval_prob_expr(e) for e in expr.elts]
                
        # Function call to allowed numeric function
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
            fname = expr.func.id
            fn = PROBABILITY_FUNCTIONS.get(fname, None)
            if fn is None:
                raise ValueError(f"Unknown probability function: {fname}")
            args = [
                self.eval_prob_expr(a) for a in expr.args
            ]
            exec_args = [
                a[0] if isinstance(a,tuple) else a for a in args
            ]

            try:
                val = fn(*exec_args)
            except Exception as e:
                raise ValueError(f"Error evaluating function '{fname}': {e}") from None
                
            arg_pointers = [
                args[k][-1] if isinstance(args[k], tuple) else self.get_value(a) for k, a in enumerate(expr.args)
            ]
            ### Keep track of numeric functions needed to compute this variable value with abstract variable arguments (if exist, if just numeric, skip), 
            # e.g., in `P2 =~ Decision(s2, Mean(P3.prob, P4.prob))` store P2 => (Mean, (P3, P4)) for later use during symbolic execution.
            if any(isinstance(a, ast.Attribute) for a in expr.args): 
                ### need to generalize 
                self.numeric_funcs[var_name] = (fn, [
                    _dep_var_name(v) if isinstance(v,str) else v for v in arg_pointers
                ], index)

            if index == -1: 
                return float(val)
            return val[index]

        # Reference to another decision's probability, e.g., `P1.prob` or `P1.p`
        if isinstance(expr, ast.Attribute) and (expr.attr == "prob" or expr.attr == "p") and isinstance(expr.value, ast.Name):
            ref_name = expr.value.id
            if ref_name not in self.decision_assignments:
                raise ValueError(f"Unknown decision variable in probability reference: {ref_name}")
            ref_prob = self.decision_assignments[ref_name][0]
            if not isinstance(ref_prob, (int, float)):
                raise ValueError(f"Referenced probability for '{ref_name}' is non-numeric: {ref_prob}")
            return float(ref_prob)
        
        ## reference to second probability in a decision, e.g., `P1.prob2` or `P1.p2`
        if isinstance(expr, ast.Attribute) and re.search(r'p(rob)?\d+', expr.attr) and isinstance(expr.value, ast.Name):
            pid = int(expr.attr[4:]) if 'prob' in expr.attr else int(expr.attr[1:])
            ref_name = expr.value.id
            if ref_name not in self.decision_assignments:
                raise ValueError(f"Unknown decision variable in probability reference: {ref_name}")
            ref_prob = self.decision_assignments[ref_name][-1]
            if not ref_prob or len(ref_prob) < pid - 1:
                raise ValueError(f"Referenced probability index {pid} out of range for variable '{ref_name}' with probabilities {ref_prob}")
            return float(ref_prob[pid - 2])
        
        ## add mean attribute reference for backward compatibility, e.g., `P1.mean` or `P1.m` to reference the mean of the probabilities in a categorical distribution
        if isinstance(expr, ast.Attribute) and expr.attr in ATTR_FUNCTIONS and isinstance(expr.value, ast.Name):
            afunc = ATTR_FUNCTIONS[expr.attr]
            ref_name = expr.value.id
            if ref_name not in self.decision_assignments:
                raise ValueError(f"Unknown decision variable in probability reference: {ref_name}")
            ref_prob = [self.decision_assignments[ref_name][0]] + self.decision_assignments[ref_name][-1] 
            ## NOTE: must match the aux-variable naming in sem.py::_init_manager.
            arg_names = [ref_name] + [f"{ref_name}__{k+2}" for k,v in enumerate(self.decision_assignments[ref_name][-1])]
            if not all(isinstance(p, (int, float)) for p in ref_prob):
                raise ValueError(f"Referenced probabilities for '{ref_name}' contain non-numeric values: {ref_prob}")
            
            return (float(afunc(*ref_prob)), (afunc, arg_names, -1))
        
        # if isinstance(expr, ast.Attribute) and (expr.attr == "mean") and isinstance(expr.value, ast.Name):
        #     ref_name = expr.value.id
        #     if ref_name not in self.decision_assignments:
        #         raise ValueError(f"Unknown decision variable in probability reference: {ref_name}")
        #     ref_prob = [self.decision_assignments[ref_name][0]] + self.decision_assignments[ref_name][-1]
        #     arg_names = [ref_name] + [f"{ref_name}{k+2}" for k,v in enumerate(self.decision_assignments[ref_name][-1])]
        #     return (float(np.mean(ref_prob)),(Mean, arg_names,-1))

        # Fallback: unsupported expression
        raise ValueError(f"Unsupported probability expression: {ast.unparse(expr)}")

            
    def visit_AnnAssign(self, node):
        """Records string variables ans decision assignments.

        Raises:
            ValueError when unknown string variable is defined in
                a decision assignment or when an assignment is defined twice.
        """
        type_ann = node.annotation.id
        var_name = node.target.id

        ## decision assignments
        if (
            type_ann == "bool"
            and isinstance(node.value, ast.Call)
            and node.value.func.id in BERNOULLI_NAMES
            and len(node.value.args) >= 2
        ):
            str_val = self.get_value(node.value.args[0])
            prob = self.eval_prob_expr(node.value.args[1], var_name)
            # When the second arg is a bare attribute reference like
            # `P25.mean` (not wrapped in a function call), eval_prob_expr
            # returns a (float_value, (afunc, arg_names, idx)) tuple — the
            # second element is for symbolic-execution bookkeeping. We
            # only want the numeric value here. The function-call branch
            # above already unwraps to a float, so we just need to handle
            # the tuple case from the attribute branches.
            if isinstance(prob, tuple):
                prob = prob[0]
            if len(node.value.args) > 2:
                prob2 = self.eval_prob_expr(node.value.args[2], var_name)
                if isinstance(prob2, tuple):
                    prob2 = prob2[0]
                prob2 = [prob2] if not isinstance(prob2, list) else prob2
            else:
                prob2 = []

            if (
                var_name in self.decision_assignments
            ):
                raise ValueError(
                    f"Decision variable defined multiple times: {var_name} in line: {ast.unparse(node)}. Please change name"
                )
            var_ref = str_val if str_val in self.string_vars else None
            str_ref = self.string_vars[str_val] if str_val in self.string_vars else str_val
            self.decision_assignments[var_name] = (prob, var_ref, str_ref, prob2)

        ## string assignments
        elif type_ann == "str" and isinstance(node.value, ast.Constant):
            str_value = self.get_value(node.value)
            if var_name in self.string_vars or var_name in self.decision_assignments:
                raise ValueError(
                    f"String variable already defined, must rename: {var_name}"
                )

            self.string_vars[var_name] = str_value
        else:
            raise ValueError(f"Unparseable line: {ast.unparse(node)}")

    def visit_Return(self, node: ast.Return) -> None:
        """Extracts return values and creates their associated variables."""
        if isinstance(node.value, ast.Constant):
            raw = str(node.value.value)
            answer_name = f"A_{str(node.value.value).replace(' ','_')}"
            
        elif isinstance(node.value, ast.Name):
            raw = str(node.value.id)
            answer_name = f"A_{node.value.id}"

        else:
            raise ValueError(f"Unknown return node: {ast.unparse(node)}")

        self.return_vals[raw] = answer_name


class ProbRefRewriter(ast.NodeTransformer):
    """Rewrites attribute references like `<Var>.prob` to numeric constants.

    This enables runtime execution of programs that use `<DecisionVar>.prob`
    inside probability expressions by replacing them with the collected
    probabilities from `VarCollector`.
    """
    def __init__(self, decisions: Dict[str, Tuple[float, Optional[str], str]]):
        super().__init__()
        self._decisions = decisions

    def visit_Attribute(self, node: ast.Attribute):
        node = self.generic_visit(node)

        if (node.attr == "prob" or node.attr == "p") and isinstance(node.value, ast.Name):
            ref = node.value.id
            details = self._decisions.get(ref, None)
            if details is None:
                return node
            prob = details[0]
            return ast.copy_location(ast.Constant(value=float(prob)), node)
        
        ## second probability reference for more complex decisions, e.g., `P1.prob2` or `P1.p2`
        if re.search(r'p(rob)?\d+', node.attr) and isinstance(node.value, ast.Name):
            pid = int(node.attr[4:]) if 'prob' in node.attr else int(node.attr[1:])
            ref = node.value.id
            details = self._decisions.get(ref, None)
            if details is None:
                return node
            ref_prob = details[-1]
            if not ref_prob or len(ref_prob) < pid - 1:
                raise ValueError(f"Referenced probability index {pid} out of range for variable '{ref_prob}")
            return ast.copy_location(ast.Constant(value=float(ref_prob[pid - 2])), node)
        
        ### add mean attribute reference for backward compatibility, e.g., `P1.mean` or `P1.m` to reference the mean of the probabilities in a categorical distribution
        if node.attr in ATTR_FUNCTIONS and isinstance(node.value, ast.Name):
            afunc = ATTR_FUNCTIONS[node.attr]
            ref = node.value.id
            details = self._decisions.get(ref, None)
            if details is None:
                return node
            ref_prob = [details[0]] + details[-1]
            return ast.copy_location(ast.Constant(value=float(afunc(*ref_prob))), node)
        
        return node
    
class ProbIndexRewriter(ast.NodeTransformer):
    """Rewrites variables to use other weihghts. Intended to be a patch for inference, which produces a new tree that 
    then gets read by the `ProgramToSemantics` visitor. 
    """
    def __init__(self, weight_id: int):
        super().__init__()
        self.weight_id = weight_id
        self._renames: Dict[str, str] = {}

    def visit_Name(self, node: ast.Name) -> ast.Name:
        new_id = self._renames.get(node.id, node.id)
        if new_id != node.id:
            return ast.copy_location(ast.Name(id=new_id, ctx=node.ctx), node)
        return node
        
    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        new_value = self.visit(node.value)  # renames base via visit_Name
        if (self.weight_id > 1 and
                isinstance(node.value, ast.Name) and
                node.value.id in self._renames and
                node.attr in ("prob", "p")):
            return ast.copy_location(
                ast.Attribute(value=new_value, attr=f"p{self.weight_id}", ctx=node.ctx),
                node,
            )
        return ast.copy_location(
            ast.Attribute(value=new_value, attr=node.attr, ctx=node.ctx),
            node,
        )
    

    

class ProgramToSemantics(ast.NodeVisitor):
    """Translation of transformed CoT program into semantic representations with
    semantic checks, and details of defined variables.

    Note: we assume (for now) a rather constrained propositional language with the
    following constraints:

        -- All probabilistic boolean variables are unique (i.e., cannot be sampled multiple times).
        -- No assignmetns are possible outside of strings (whcih are not factored into the semantics).
        -- All `If-then` conditions have an Else-cond that raises an exception.

    Attributes:
        guard_stack: An auxiliary stack datastructure for computing variable guards (i.e., the
            Boolean conditions determining the execution of variables).
        guards: Probabilistic booleans with their guards.
        evidence: Evidence or observe statements to be used for conditioning. 

    """

    def __init__(
        self,
        mapper: str = "sympy",
        string_assignments: Optional[Dict] = None,
        decision_assignments: Optional[Dict] = None,
        return_vars: Optional[Dict] = None,
        var_list: List[str] = None,
    ) -> None:
        """Initializes semantic translator.

        Arguments:
            mapper: The particular mapping function being used. 
            string_assignments: The string propositions provided in the program.
            decision_assignments: A record of the decision variables and lines.
            evidence: Evidence or observe statements to be used for conditioning. 
        """
        if string_assignments is None:
            string_assignments = {}
        if decision_assignments is None:
            decision_assignments = {}
        if return_vars is None:
            return_vars = {}
        if var_list is None:
            var_list = []
        
        self.mapper = BuildMapper(
            mapper,
            var_list=var_list,
            decision_vars=decision_assignments,
            return_vars=return_vars,
        )
        self.guards = {}
        self.string_assignments = string_assignments
        self.decision_assignments = decision_assignments
        self.return_variables = return_vars 
        self.evidence = []
        self.decisions_used_in_conds: set[str] = set()

    def extract_cond(self, test, relax: Optional[bool] = False):
        """Extracts Boolean formulas from conditionals

        Returns:
            Boolean sympy formula.
        """
        return extract_cond(
            self.mapper, 
            test=test,
            bools=self.decision_assignments,
            relax=relax,
        )

    def _select_body(self, node: ast.Module):
        """Choose the particular part of the node to analyze. Sometimes this might
        contain top-level functions, which are safe to ignore.

        Arguments:
            node: The incoming node and module to create semantics from.
        Raises:
            ValueError when multiple functions are encountered in input module.

        """
        funcs = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        if len(funcs) == 0:
            return node.body
        if len(funcs) == 1:
            return funcs[0].body

        names = [f.name for f in funcs]
        raise ValueError(
            f"Multiple functions found in module: {names}. "
            "Please provide a module with a single function or preselect one before analysis."
        )

    def visit_Module(self, node: ast.Module) -> None:
        """Visits all modules. Starts by first collecting all variables, then
        extracts guards and formulas.

        Arguments:
            node: The incoming module node from ast..
        """
        body = self._select_body(node)
        self.visit_block(body, self.mapper.true, finished=False)

    def visit_block(self, stmts, in_guard, finished: bool):
        """Visit a sequence of statements under a particular guard.

        Arguments:
            stms: The input statement(s).
            in_guard: The particular guard in scope.
            finished: Boolean switch to determine if the end of
                the program has been detected.

        Returns:
             A formula produced by the instantiated symbolic mapper.
        """
        g = in_guard
        for stmt in stmts:
            g, finished = self.visit_stmt(stmt, g, finished)
        return g

    def _handle_observe(self,call: ast.Call, in_guard) -> bool:
        """Handles the observes and gets out the conditioning formulas. 

        Raises: 
            ValueError when observe is not in global context (nested observes not
                yet implemented). 
        Returns: 
            `True` if an observe call is found, `False` otherwise. 

        """
        if isinstance(call.func, ast.Name) and call.func.id in set(["observe","evidence"]):
            if not (in_guard == self.mapper.true or in_guard == self.mapper.false):
                raise ValueError('Observe not allowed in nested context')

            evidence = self.extract_cond(call.args[0], relax=True)
            self.evidence.append(evidence)
            return True
        
        return False
    
    def visit_stmt(self, node: ast.expr, in_guard, finished: bool):
        """Visits an individual statement within a particular guard. Note that this
        method enumerates exhaustively all possible syntactic forms in the languages
        and raises an Exception when an unknown form is encountered.

        Arguments:
            node: The input node in the tree.
            in_guard: The overarching guard in this statement.
            finished: Boolean switch to determine if the end of the code has been reached.

        Raises:
            ValueError when unknown node is encountered. 

        """
        def extract_names(target) -> list[str]:
            if isinstance(target, ast.Name):
                return [target.id]
            elif isinstance(target, ast.Tuple):
                names = []
                for elt in target.elts:
                    names.extend(extract_names(elt))
                return names
            elif isinstance(target, ast.Starred):
                return extract_names(target.value)
            else:
                raise ValueError(f"Unsupported assignment target: {ast.dump(target)}")
        func_call = isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        
        if finished:
            if func_call:
                call = node.value
                self._handle_observe(call, in_guard)
                
            return self.mapper.false, True

        if func_call:
            call = node.value
            is_observe = self._handle_observe(call, in_guard)

            if is_observe is True: 
                return in_guard, False

        if isinstance(node, ast.AnnAssign):
            if isinstance(node.annotation, ast.Name) and node.annotation.id == "bool":
                var = self.mapper.Symbol(node.target.id)
                self.guards.setdefault(var, []).append(in_guard)
            return in_guard, False

        if isinstance(node, ast.Assign):
            ## want to pull out the left hand side target variables
            vars = []
            for target in node.targets:
                vars.extend(extract_names(target))
            for var in vars: 
                var_sym = self.mapper.Symbol(var)
                self.guards.setdefault(var_sym, []).append(in_guard)

            if len(vars) < 2:
                raise ValueError(
                    'Only categorial variable assignment allowed here with tuples'
                )    

            # ## add mutual exclusivity to evidence for categorical variables, e.g., if `X, Y = Categorical(...)` then add `X => ~Y` and `Y => ~X` to evidence.
            total_or = self.mapper.false
            for i in range(len(vars)):
                total_or = self.mapper.Or(total_or, self.mapper.Symbol(vars[i], return_val=True))
                for j in range(i+1, len(vars)):
                    var_i_sym = self.mapper.Symbol(vars[i], return_val=True)
                    var_j_sym = self.mapper.Symbol(vars[j], return_val=True)
                    self.evidence.append(
                        self.mapper.Or(
                            self.mapper.Not(var_i_sym),
                            self.mapper.Not(var_j_sym)
                        )
                    )
                    
            # # one and only one of the variables can be true
            self.evidence.append(total_or)

            return in_guard, False 
                
        if isinstance(node, ast.Import):
            raise ValueError(f"Import not allowed: {ast.unparse(node)}")

        if isinstance(node, ast.If):
            cond = self.extract_cond(node.test)

            # if part of condition
            then_in = self.mapper.And(in_guard, cond)
            then_out = self.visit_block(node.body, then_in, finished=False)
            
            # else part of condition
            if node.orelse:
                else_in = self.mapper.And(in_guard, self.mapper.Not(cond))
                else_out = self.visit_block(node.orelse, else_in, finished=False)
            else:
                else_out = self.mapper.And(in_guard, self.mapper.Not(cond))

            output = self.mapper.Or(then_out, else_out)
            return output, (output is self.mapper.false)

        if isinstance(node, ast.Return):

            ## for returning variable
            if isinstance(node.value, ast.Constant):
                answer_name = f"A_{str(node.value.value).replace(' ','_')}"
            elif isinstance(node.value, ast.Name):
                answer_name = f"A_{node.value.id}"

            var = self.mapper.Symbol(answer_name)

            #self.return_variables.add(var)
            self.guards.setdefault(var, []).append(in_guard)
            return self.mapper.false, True

        if isinstance(node, ast.Raise):
            var = self.mapper.Symbol("Exc")
            self.guards.setdefault(var, []).append(in_guard)
            return self.mapper.false, True

        if isinstance(node, ast.Pass):
            return in_guard, False

        if isinstance(node, ast.Assert):
            ## consider making assertions a special distinct exception variable
            assert_test = node.test
            assert_cond = self.extract_cond(assert_test)

            var = self.mapper.Symbol("Exc")
            self.guards.setdefault(var, []).append(
                self.mapper.And(in_guard, self.mapper.Not(assert_cond))
            )

            return self.mapper.And(in_guard, assert_cond), False

        ##TODO: add more constructs here.
        raise ValueError(f"Unsupported statement: {ast.dump(node)}")

    def compute_semantics(self):
        """Obtains a symbolic rendering of the Boolean program
        based on the computed guards.

        Returns:
            A formula constructed via the instantiated symbolic mapper.

        """
        return self.mapper.construct_formula(
            self.guards,
            self.decision_assignments,
            self.evidence,
        )
    
    def symbolic_formula(self):
        """Obtains a symbolic rendering of the Boolean program
        based on the computed guards.

        Returns:
            A formula constructed via the instantiated symbolic mapper.

        """
        return self.mapper.construct_formula(
            self.guards,
            self.decision_assignments,
            self.evidence,
            #decision_uses=self.decisions_used_in_conds,  # NEW
        )
    


class CoTProgramTransformer(ast.NodeTransformer):
    """Node transformer for the CoT custom language.

    Attributes
        decision_strings: A dictionary of decision variables and the
            string values that they correspond to.

    """

    def __init__(self) -> None:
        self.decision_strings = dict()
        super().__init__()

    def infer_type(self, value: Union[ast.Call, ast.Constant], name: str) -> str:
        """Infer type for any input assignment. Also returns the string or
            variable associated with a decision line.

        Arguments:
            value: The value of an assignment (either annotated assignment
                or non annotated assignment).
            name: The name of the variable in the assignment.
        Raises:
            TypeError when unknown (non `str` or `bool`) type is encountered.

        """

        def get_decision_node(arg: Union[ast.Constant, ast.Name]) -> Tuple[str, str]:
            """Finds the decision variable value and the type of that variable.

            Arguments:
                arg: The incoming ast node.
            Returns:
                A tuple of the value for that decision and the type (e.g., context variable or raw
                    string value).
            """
            if isinstance(arg, ast.Constant):
                return (arg.value, "value")
            elif isinstance(arg, ast.Name):
                return (arg.id, "name")
            raise ValueError(f"Unknown decision value: {ast.unparse(arg)}")

        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return "str"
        if isinstance(value, ast.Constant) and isinstance(value.value, bool):
            return "bool"
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in BERNOULLI_NAMES
        ):
            self.decision_strings[name] = get_decision_node(value.args[0])
            return "bool"
        raise ValueError(
            f"Unknown type encountered in assignment: {ast.unparse(value)}, lineno={value.lineno}"
        )


    ####################
    # ### assignments  #
    ####################

    def visit_Assign(self, node: ast.Assign):
        """Translates all assignments to type annotated assignments, and performs type checking.

        Raises:
            TypeError whenever multiple assignemtns are used.

        """
        value = node.value
        prob_assignment = False 
        if isinstance(node.value, ast.UnaryOp) and isinstance(
            node.value.op, ast.Invert
        ):
            value = value.operand
            prob_assignment = True 

        target = node.targets[0]

        ## single value assignment 
        if not isinstance(target, ast.Tuple): 

            var_name = target.id
            inferred_type = self.infer_type(value, var_name)
            if inferred_type != "bool" and prob_assignment:
                raise ValueError('Only booleans can have probabilistic assignment')
        
            annotation = ast.Name(id=inferred_type, ctx=ast.Load())
        
            annassign = ast.AnnAssign(
                target=target,
                annotation=annotation,
                value=value,
                simple=1,
            )
            return ast.copy_location(annassign, node)
        
        ## tuple assignment (for categoricals) without ~
        else: 
            assign = ast.Assign(
                targets=[target],
                value=value,
                lineno=node.lineno,
                col_offset=node.col_offset,
            )
            return ast.copy_location(assign, node)
        

    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Perform type checking on annotated assignments.

        Raises:
            TypeError when an unknown or mismatched type is encountered.

        """
        prob_assignment = False 
        if (
            node.value
            and isinstance(node.value, ast.UnaryOp)
            and isinstance(node.value.op, ast.Invert)
        ):
            node.value = node.value.operand
            prob_assignment = True 

        type_ann = node.annotation.id
        var_name = node.target.id
        infer_type = self.infer_type(node.value, var_name)
        if infer_type != "bool" and prob_assignment:
            raise ValueError('Only booleans can have probabilistic assignment')

        if type_ann not in set(["str", "bool"]) or type_ann != infer_type:
            raise TypeError(
                f"Encountered unknown/wrong type: {type_ann}, inferred={infer_type}, lineno={node.lineno}"
            )
        return self.generic_visit(node)

    # ##################
    # # return values  #
    # ##################

    def visit_Return(self, node: ast.Return):
        """Visits return values and ensures that it returns a concrete value. Also
        implements a small short-cut for returning values of decision variables.

        For example, in the program:

            ```
            P = Decision(s, 0.5)
            if P:
                return P.answer ## also can be `P.string`
            else:
                raise ValueError(s)
            ```

        this gets expanded out to:

            ```
            P = Decision(s, 0.5)
            if P:
                return s
            else:
                raise ValueError(s)
            ```

        Raises:
            ValueError when variable is not referenced.

        """
        ### decision short-cut macros
        val = node.value
        if (
            isinstance(val, ast.Attribute)
            and val.attr in set(["string", "answer"])
            and isinstance(val.value, ast.Name)
        ):
            name = val.value.id

            try:
                str_val, rtype = self.decision_strings[name]

            except KeyError:
                raise ValueError(f"Returning unknown variable: {name}")

            return ast.copy_location(
                ast.Return(self._build_return(node, str_val, rtype)), node
            )

        if not (
            isinstance(node.value, ast.Constant) or isinstance(node.value, ast.Name)
        ):
            raise ValueError(
                f"Can only return variable or constant, invalid line: `{ast.unparse(node)}`"
            )
        return self.generic_visit(node)

    def _build_return(self, node, str_val, rtype):
        """Builds the return value for a given yield value or shorthand value.

        Arguments:
            node: The node where the return value is.
            str_val: The value of the return
            rtype: The particular type of val it is.

        """
        if rtype == "name":
            n = ast.Name(id=str_val, ctx=ast.Load())
        elif rtype == "value":
            n = ast.Constant(value=str_val)
        else:
            raise ValueError(f"Unknown value type: {rtype}")
        return ast.copy_location(n, node)

    def visit_Expr(self, node: ast.Expr):
        """Visits general expressions and re-interprets the meaning of `yield` to be if-then.

        `yield` in particular gets translated into a if-then return condition. E.g., in the
        program below:

            ```
            P =~ Decision(s, 0.5)
            yield P
            ```

        Gets translated into the following in native Python:

            ```
            P = Decision(s, 0.5)
            if P:
                return s
            else:
                raise ValueError(s)
            ```

        Raises:
            SyntaxError when yields are empty. ValueError when the yield
                does not return a known variable.

        """
        node = self.generic_visit(node)

        if isinstance(node.value, ast.Yield):
            y = node.value
            if y is None:
                raise SyntaxError("Yield must return a value")
            try:
                yield_var = y.value.id
                str_val, rtype = self.decision_strings[yield_var]

            except AttributeError:
                raise ValueError("Yield must return a decision variable")
            except KeyError:
                raise ValueError("Yield must return a known decision variable")
            except Exception:
                raise ValueError("Issue parsing yield statement")

            if_node = ast.If(
                test=y.value,
                body=[ast.Return(self._build_return(node, str_val, rtype))],
                orelse=[
                    ast.Raise(
                        exc=ast.Call(
                            func=ast.Name(id="ValueError", ctx=ast.Load()),
                            args=[
                                ast.Constant(
                                    value=f"Cannot establish answer/return via {yield_var}"
                                )
                            ],
                            keywords=[],
                        ),
                        cause=None,
                    )
                ],
            )
            return ast.copy_location(if_node, node)
        
        return node

    def visit_Import(self, node: ast.Import): 
        raise ValueError(f"Import not allowed: {ast.unparse(node)}")

    def visit_Lambda(self, node: ast.Lambda):
        raise ValueError(f"Unsupported statement: {ast.dump(node)}")
