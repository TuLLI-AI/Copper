import math
import json
import numpy as np
import re
import sys
import argparse
import logging
from argparse import Namespace
from datasets import load_dataset
import traceback
from tqdm import tqdm
from openai import OpenAI
from typing import List, Dict, Optional, Union, Tuple, Set, Any
from openai import OpenAI
from together.client import Together
from openai.types.chat.chat_completion import Choice
from together.types.chat_completions import ChatCompletionChoicesData
from cot_prog.prompt.cot_text_parser import CoTContreteSyntaxParser
from cot_prog.prompt.prompts import (
    USER_TEMPLATE,
    GENERIC_COT_INSTRUCTION_BINARY,
    MULTI_ANSWER_BINARY,
)

LLMType = Union[Together, OpenAI]
LLMOutput = Union[Choice, ChatCompletionChoicesData]

INSTRUCTIONS = {
    "generic"      : GENERIC_COT_INSTRUCTION_BINARY,
    "vanilla"      : GENERIC_COT_INSTRUCTION_BINARY,
    "multi_answer" : MULTI_ANSWER_BINARY,
}

def get_model(client: Optional[str] = "openai") -> LLMType:
    """Builds the LLM client based on specified client (currently limited to
    only openai and together clients).

    Arguments:
        config: The global configuration.

    Returns:
        The underlying LLM client.
    """
    if "together" in client:
        client = "together"
    else:
        client = "openai"
    
    llm = OpenAI() if client == "openai" else Together()
    return llm

def find_logps(
    content: LLMOutput,
    steps, 
    *,
    remove_meta_tokens: Optional[bool] = True,
    length_normalization: Optional[bool] = True,
) -> List[float]:
    """Find the subsequence log probabilities. If `remove_meta_tokens` is 
    set to True it will skip certain keynote tokens (e.g., `Based on <nums>`) and 
    if `length_normalization` is set to true, it will compute a geometric mean 
    of the token probabilities (rather than just a product). 

    Arguments: 
         content: The full model output. 
         remove_meta_tokens: Switch to remove language keywords when 
             computing the sub-sequence probabilities. 
         length_normalization: Use length normalization to obtain the final 
             sub-sequence probabilities (take geometric mean of probs). 

    Raises: 
        An assertion error when the steps and logps splits don't add up.

    """
    
    def get_steps(logp_list: List[Any]) -> List[Tuple[str,float,str]]:
        """Splits the log probability list into steps based on newlines."""
        token_list = []
        seq_scores = []
        full_string = ""
        
        for token in logp_list:
            t = token.token if hasattr(token, "token") else token['token']
            logp = token.logprob if hasattr(token, "logprob") else token['logprob']
            newline = re.search("\n+", t)
            full_string += t
            
            if newline:
                seq_scores.append(token_list)
                token_list = []
                full_string = ""
                continue
            
            token_list.append((t, logp, full_string))    

        if token_list:
            seq_scores.append(token_list)

        

        return seq_scores 
            
    ## openai models, more recent together models. 
    if hasattr(content.logprobs, "content"):
        logp_list = content.logprobs.content
    ## older together models 
    else:
        raw_tokens = content.logprobs.tokens
        raw_logps = content.logprobs.token_logprobs
        assert len(raw_tokens) == len(raw_logps)
        logp_list = [
            {"token" : raw_tokens[i], "logprob" : raw_logps[i]} for i in range(len(raw_tokens))
        ]

    step_scores = get_steps(logp_list)
    assert len(step_scores) == len(steps), f"Error with logp alignment: {len(step_scores)} steps vs {len(steps)} expected steps. Please check the formatting of the output and the logprobs content."

    final_probs = []
    for k,seq in enumerate(step_scores):
        logp_list = []
        pstep = steps[k]
        step_str = pstep.raw
        sindex   = pstep.start_index
        eindex   = pstep.end_index

        sstring = step_str[:eindex] if eindex > -1 and remove_meta_tokens \
          else step_str
        prefix = step_str[:sindex] if sindex > 0 and remove_meta_tokens \
          else "EMPTY"

        seq_probs = []
        for (token, logp, full_string) in seq:
            ## sometimes this comes up in 
            if logp == -9999.0: continue 
            if full_string.strip() == sstring.strip():
                break
            if full_string.strip() == prefix.strip():
                continue
            seq_probs.append(logp)

        seq_score = np.sum(np.clip(seq_probs, a_min=-20.0, a_max=None))
        if length_normalization:
            seq_score = (1 / len(seq_probs)) * seq_score

        final_probs.append(round(float(np.exp(seq_score)),3))
        
    return final_probs

def prompt_to_program(
    query: Union[str, dict],
    query_field: Optional[str] = "question",
    llm: Optional[LLMType] = None,
    model_name: Optional[str] = 'gpt-4o',
    client: Optional[str] = None,
    num_retries: Optional[int] = 5,
    instruction: Optional[str] = "generic",
    debug: Optional[bool] = False,
    user_format: Optional[str] = None,
    remove_meta_tokens: Optional[bool] = True,
    length_normalization: Optional[bool] = True,
    temperature: Optional[float] = 0.0,
    top_p: Optional[float] = None,
    strict: Optional[bool] = True,
    **kwargs,
    ):
    """Prompts for Chain-of-Thought (CoT) then checks it and formats according to 
    a probabilistic program. 

    Arguments: 
        query: The underlying query to the language model. 
        query_field: The field where the query sits (if the query is inside of 
            a dictionary object). 
        llm: The (optional) LLM client to query/ 
        model_name: The name of the model to query. 
        client: The (optional) name of the client to use (currently limited to openai
            and together.ai).
        num_retries: The number of retries to get well-formed probabilistic program.
        instruction: The system prompt (or pointer to defaults defined here). 
        debug: Print intermediate system output. 
        user_format: (optional) Formatting for the user input.  
        remove_meta_tokens: Switch to determine whether or not to remove meta-tokens 
            (e.g., `based on X`) when computing sub-sequence probabilities. 
        length_normalization: Switch to use length normalization when getting sub-sequence 
            probability information. 
        temperature: The standard temperature parameter for decoding (default to 0.0). 
        top_p: Standard nucleas sampling decoding parameter (default to None). 
        strict: Whether to use strict rules when checking the validity of the resulting 
            chain of thought programs. (defaults to True)

    """
    # Dispatch to multi_answer_cot pipeline if requested
    if instruction == "multi_answer_cot":
        from cot_prog.prompt.multi_answer_cot import multi_answer_cot_to_program
        return multi_answer_cot_to_program(
            query=query, query_field=query_field, llm=llm,
            model_name=model_name, client=client, num_retries=num_retries,
            debug=debug, remove_meta_tokens=remove_meta_tokens,
            length_normalization=length_normalization, temperature=temperature,
            top_p=top_p, strict=strict, **kwargs,
        )

    orig_query = query
    llm = llm if llm is not None else get_model(client=(client or model_name))
    query = query[query_field] if isinstance(query, dict) else query
    if instruction in INSTRUCTIONS and user_format is None:
        query = f"Question: {query}\nAnswer (step-by-step):\n"

    error_msg = "No errors, first attempt"
    attempts = 0
    parsing_errors = []
    cot_program = None 

    while True:
        attempts += 1
        if attempts > num_retries:
            break

        system_prompt = INSTRUCTIONS.get(instruction,None) or instruction
        system_prompt = system_prompt.format(errors=error_msg)
        messages = [
            {
                "role" : "system",
                "content" : [{
                    "type" : "text",
                    "text" : system_prompt,
                }]
            },
            {
                "role" : "user",
                "content" : [{
                    "type" : "text",
                    "text" : query,
                }]
            }
        ]
        response = llm.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=messages,
            top_p=top_p,
            store=True,
            logprobs=True,
        )

        raw_message = response.choices[0].message.content
        cot_parser = CoTContreteSyntaxParser()
        if debug:
            print(system_prompt)
            print(query)
            print(raw_message)
            print("========================")

        try: 
            cot_program = cot_parser.parse(raw_message, query=orig_query)
            ## check structure and integrity of the program
            program_error = cot_program.check_program(strict=strict)
            if program_error:
                raise ValueError(program_error)
                    
        except Exception as e:
            error_msg = str(e)
            parsing_errors.append(error_msg)
            continue

        cot_program.check_program(strict=strict)

        step_probs = find_logps(
            response.choices[0],
            cot_program.lines,
            remove_meta_tokens=remove_meta_tokens,
            length_normalization=length_normalization
        )
        cot_program.add_weights(step_probs)
        cot_program.attempts = attempts
        cot_program.errors = parsing_errors 
        if debug:
            program = cot_program.write_program()
            print("PROGRAM")
            print(program)
            
        return cot_program
