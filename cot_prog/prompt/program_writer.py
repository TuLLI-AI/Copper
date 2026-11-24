__all__ = ["write_program"]

### 
SINGLE_ANSWER_TEMPLATE = """
### original query 
query = "{query}"

### step strings 
{string_vars}

### step propositions
{prop_vars}

### step relations 
{conds}
"""

MULTI_ANSWER_TEMPLATE = """
### original query 
query = "{query}"

### step strings 
{string_vars}

### step propositions
{prop_vars}

### step assertions 
{conds}

## answer branches
{answer_conds}
"""

MULTI_ANSWER_TEMPLATE_SIMPLE = """
### original query 
query = "{query}"

### step strings 
{string_vars}

### step propositions
{prop_vars}

## answer branches
{answer_conds}
"""

COND  = "if {formula}:\n    {prop_var}\nelse:\n    raise ValueError(f'Cannot establish step(s): {formula}')"
ACOND = "if {formula}:\n    {prop_var}\n    yield {prop_var_name} ## answer \nelse:\n    raise ValueError(f'Cannot establish step(s): {formula}')"
ASSERT = "assert {formula}, 'Cannot establish step: step{step}'"
FIRST_ACOND = "if ({formula}) and (not {neg_formula}):\n    yield {prop_var_name} ## answer"
SEC_ACOND = "elif {formula}:\n    yield {prop_var_name} ## answer"
ACOND_EXCEPTION = "else:\n    raise ValueError('Cannot establish answers')"

FIRST_ACONDS_SIMPLE = "if ({formula}):\n    return {prop_var_name} ## answer"
SEC_ACOND_SIMPLE = "elif ({formula}):\n    return {prop_var_name} ## answer"

def write_program(query, lines) -> str:
    """Writes out probabilistic programs from CoT objects."""
    answers = [l for l in lines if l.is_answer]
    string_vars = '\n'.join([l.string_var() for l in lines])

    def get_all_premises(prefix, graph, visited=None):
            if visited is None:
                visited = set()
            if prefix in visited:
                return set()
            visited.add(prefix)
            all_premises = set(graph.get(prefix, []))
            for p in graph.get(prefix, []):
                all_premises.update(get_all_premises(p, graph, visited))
            return all_premises

    ### single answer programs 
    if len(answers) == 1:
        prop_vars = '\n'.join([l.prop_var() for l in lines if not l.premises])
        conds = []
        for line in lines:

            if line.formula:
                if not line.answer:
                    cond = COND.format(
                        formula=line.formula,
                        prop_var=line.prop_var()
                    )
                else:
                    cond = ACOND.format(
                        formula=line.formula,
                        prop_var=line.prop_var(),
                        prop_var_name=line.prop_var_name,
                    )
                conds.append(cond)

        conds = '\n'.join(conds)
        return SINGLE_ANSWER_TEMPLATE.format(
            query=query,
            string_vars=string_vars,
            prop_vars=prop_vars,
            conds=conds
        )
    
    else:
        prop_vars = '\n'.join([l.prop_var() for l in lines])
        premise_graph = {l.prefix: l.premises for l in lines}
        conds = []
        answer_conds = []
        branches = []

        
        for line in lines:
            if line.formula and line.answer:
                all_premises = get_all_premises(line.prefix, premise_graph)
                all_premises.add(line.prefix)
                bool_formula = ' and '.join([f"P{p}" for p in all_premises])

                if not branches: 
                    cond = FIRST_ACONDS_SIMPLE.format(
                        formula=bool_formula,
                        prop_var_name=f"{line.prop_var_name}.answer",
                    )
                    branches.append(cond)
                else:
                    cond = SEC_ACOND_SIMPLE.format(
                        formula=bool_formula,
                        prop_var_name=f"{line.prop_var_name}.answer",
                    )
                    branches.append(cond)
        
        branches.append(ACOND_EXCEPTION)
        branches = '\n'.join(branches)
        return MULTI_ANSWER_TEMPLATE_SIMPLE.format(
            query=query,
            string_vars=string_vars,
            prop_vars=prop_vars,
            answer_conds=branches
        )


        # for k, (formula, prop_var_name) in enumerate(branches):
        #     if k == 0:
        #         cond = FIRST_ACONDS_SIMPLE.format(
        #             formula=formula,
        #             prop_var_name=prop_var_name,
        #         )
        #         branches.append(cond)
        #     else:
        #         cond = SEC_ACOND_SIMPLE.format(
        #             formula=formula,
        #             prop_var_name=prop_var_name,
        #         )
        #         branches.append(cond)

        
        
        # ###          
        # for line in lines:
        #     if line.formula:
        #         if not line.answer:
        #             cond = ASSERT.format(formula=line.formula, step=line.prefix)
        #             conds.append(cond)
        #         else:
        #             answer_conds.append((line.formula,line.prop_var(), line.prop_var_name))

        # for k, (formula, prop_var, prop_var_name) in enumerate(answer_conds):
        #     if k == 0:
        #         nformula = ' and '.join([f"({n[0]})" for n in answer_conds[1:]])
        #         cond = FIRST_ACOND.format(
        #             formula=formula,
        #             neg_formula=nformula,
        #             prop_var=prop_var,
        #             prop_var_name=prop_var_name,
        #         )
        #         branches.append(cond)
        #     else:
        #         cond = SEC_ACOND.format(
        #             formula=formula,
        #             prop_var=prop_var,
        #             prop_var_name=prop_var_name,
        #         )
        #         branches.append(cond)
                
                
        # conds = '\n'.join(conds)
        # branches.append(ACOND_EXCEPTION)
        # branches = '\n'.join(branches)
        # return MULTI_ANSWER_TEMPLATE.format(
        #     query=query,
        #     string_vars=string_vars,
        #     conds=conds,
        #     prop_vars=prop_vars,
        #     answer_conds=branches
        # )
                    
            

        
