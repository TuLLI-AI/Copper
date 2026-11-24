from typing import Union, Callable, Dict, List, Tuple, Optional, Set
from .compiler import CoTProgParser, CotProg


def ParseStringCot(program: str, name: Optional[str] = "cot") -> CotProg:
    """Parses a CotProg as a string using the parser above.

    Arguments:
        program: The program in a string form.
        name: The name of the program (optional).
    Returns:
        `CotProg` object, representing the program.
    Raises: 
         ValueError when program string is empty. 

    """
    if not program.strip():
        raise ValueError(f'Empty program, must be non-empty: {program}')
    
    indented = "\n".join([f"    {l}" for l in program.split("\n")])
    full_program = f"def {name}():\n{indented}"
    cot_program = CoTProgParser()(full_program)
    
    return cot_program
