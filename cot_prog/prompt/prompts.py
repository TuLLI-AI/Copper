GENERIC_COT_INSTRUCTION_BINARY = """Please answer each question thinking step-by-step. Rather than giving a single answer, however, I want you to express different possible reasoning paths. 

For example: 

Question: Could the members of the Police perform lawful arrests? Answer only yes/no.
Answer (step-by-step):
1. My understanding is that the members of The Police were musicians. (ref. none)
2. Only law enforcement officers can perform lawful arrests (ref. none)
3. Based on 1 and 2, the answer is: no (ref. 1, 2) (final answer: no)

where two distinct answers are in line 3 (no) and line 5 (yes) and end in (final answer: <yes/no>)

Another example: 

Question: Is Newt Gingrich's nickname a type of Reptile?
Answer (step-by-step): 
1. Newt Gingrich was born Newton Leroy Gingrich and goes by the nickname Newt. (ref. none)
2. A newt is a type of Amphibian with smooth, sticky skin. (ref. none)
3. From what I recall, Amphibians are not reptiles. (ref. none)
4. Based on 1. 2 and 3, a newt is not a kind of reptile. (ref. 1, 2, 3)
5. Based on 4, the answer is 'no'. (ref. 4) (final answer: no)

Please only answer using this format where:

-- Every line of text must start with a number, including answer lines.
-- Every line at the end has a reference to past related facts (in parentheses (ref. <steps>)) or none. THIS STATEMENT MUST BE IN EACH LINE. Try not to reference too many steps in each line, break up your argument.
-- The contents of (ref. ...) MUST be either the literal word "none" or a comma-separated list of step numbers (e.g. "ref. 1", "ref. 1, 2, 3"). Do NOT write free-text references such as "ref. passage", "ref. question", or "ref. premise" — if a step has no prior step to cite, write "(ref. none)".
-- There should be ONE answer lines, either yes or no. These steps should end in (final answer: <yes or no>).
-- All non-answer steps should be referenced at least once (in a future (ref. <stems>)) to make your argument transparent and easy to follow.

DO NOT USE ANY OTHER FORMAT IN YOUR ANSWER. DO NOT ADD ADDITIONAL TEXT OUTSIDE OF THIS. Your output will be checked for errors and if found you will be asked to repeat (errors are marked below).

errors: {errors}

"""

# ---------------------------------------------------------------------------
# Vanilla CoT — multiple-choice
# ---------------------------------------------------------------------------
# Mirrors GENERIC_COT_INSTRUCTION_BINARY but lets the model output a
# letter A/B/C/... in (final answer: X). The choices block is concatenated
# into the user message (NOT this system prompt) by the runner.
GENERIC_COT_INSTRUCTION_MCQ = """Please answer the multiple-choice question thinking step-by-step.

For example:

Question: Three people - Alice, Bob, and Carol - finished a race. (1) Carol finished before Bob. (2) Alice did not finish first. Who finished first?
A. Alice
B. Bob
C. Carol
D. Cannot be determined
Answer (step-by-step):
1. Constraint (2) says Alice did not finish first, so Alice is not first. (ref. none)
2. Constraint (1) says Carol finished before Bob, so Bob is not first. (ref. none)
3. The only remaining option is Carol. (ref. 1, 2)
4. Based on 1, 2, 3, the answer is: C (ref. 1, 2, 3) (final answer: C)

Please only answer using this format where:

-- Every line of text starts with a number, including the answer line.
-- Every line ends with (ref. <steps>) or (ref. none). THIS STATEMENT MUST BE IN EACH LINE. Try not to reference too many steps in each line, break up your argument.
-- The contents of (ref. ...) MUST be either the literal word "none" or a comma-separated list of step numbers (e.g. "ref. 1", "ref. 1, 2, 3"). Do NOT write free-text references such as "ref. passage", "ref. question", or "ref. premise" — if a step has no prior step to cite, write "(ref. none)".
-- There is ONE answer line ending in (final answer: <one of the option letters>).
-- The letter in (final answer: X) MUST be one of the option letters shown in the question (typically a single capital letter A, B, C, ...). Do NOT write the option text — only the letter.
-- All non-answer steps should be referenced at least once.

DO NOT USE ANY OTHER FORMAT IN YOUR ANSWER. DO NOT ADD ADDITIONAL TEXT OUTSIDE OF THIS. Your output will be checked for errors and if found you will be asked to repeat (errors are marked below).

errors: {errors}

"""

MULTI_ANSWER_BINARY = """Please answer each question thinking step-by-step. Rather than giving a single answer, however, I want you to express different possible reasoning paths. 

For example: 

Question: Could the members of the Police perform lawful arrests? Answer only yes/no.
Answer (step-by-step): 
1. My understanding is that the members of The Police were musicians.
2. Only law enforcement officers can perform lawful arrests.
3. Based on 1 and 2, the answer is: no (ref. 1, 2) (final answer: no)
4. One could alternatively interpret 'The Police' to be a group of real police.
5. Alternatively, based on 2 and 4, the answer is: yes  (ref. 2, 4) (final answer: yes)

where two distinct answers are in line 3 (no) and line 5 (yes) and end in (final answer: <yes/no>)

Another example: 

Question: Is Newt Gingrich's nickname a type of Reptile?
Answer (step-by-step): 
1. Newt Gingrich was born Newton Leroy Gingrich and goes by the nickname Newt. (ref. none)
2. A newt is a type of Amphibian with smooth, sticky skin (ref. none)
3. Amphibians often look like common reptiles, so they might be reptiles. (ref. none)
4. Based on 1, 2 and 3, the answer is 'yes' (ref. 1, 2, 3) (final answer: yes)
5. From what I recall now, it might be that Amphibians are not reptiles. (ref. none)
6. Considering 1, 2 and 5, a newt is not a kind of reptile. (ref. none)
7. Based on this, the answer is 'no'. (ref. 1, 2, 5, 6) (final answer: no)

Please only answer using this format where: 

-- Every line of text must start with a number, including answer lines. 
-- Every line at the end has a reference to past related facts (in parentheses (ref. <steps>)) or none. THIS STATEMENT MUST BE IN EACH LINE. Try not to reference too many steps in each line, break up your argument. ONLY ANSWER LINES SHOULD REFERENCE OTHER STEPS. NON-ANSWER STEPS SHOULD NOT REFERENCE OTHER STEPS.
-- There should be TWO answer lines, one for 'yes' and one for 'no'. These steps should end in (final answer: <yes or no>).  
-- All non-answer steps should be referenced at least once (in a future (ref. <stems>)) to make your argument transparent and easy to follow. 

DO NOT USE ANY OTHER FORMAT IN YOUR ANSWER. DO NOT ADD ADDITIONAL TEXT OUTSIDE OF THIS. Your output will be checked for errors and if found you will be asked to repeat (errors are marked below). 

errors: {errors}

"""

USER_TEMPLATE = """Question: {query}
Target answer: {candidate}
Answer (step-by-step):
"""

DISCOVERY_COT_INSTRUCTION = """Please answer each question thinking step-by-step. Rather than giving a single answer, however, I want you to express different possible reasoning paths.

For example:

Question: Could the members of the Police perform lawful arrests?
Answer (step-by-step):
1. My understanding is that the members of The Police were musicians. (ref. none)
2. Only law enforcement officers can perform lawful arrests (ref. none)
3. Based on 1 and 2, the answer is: no (ref. 1, 2) (final answer: no)

Another example:

Question: What is the largest planet in the solar system?
Answer (step-by-step):
1. The solar system has eight planets. (ref. none)
2. Jupiter is the largest planet by mass and diameter. (ref. none)
3. Based on 1 and 2, the answer is Jupiter. (ref. 1, 2) (final answer: Jupiter)

Please only answer using this format where:

-- Every line of text must start with a number, including answer lines.
-- Every line at the end has a reference to past related facts (in parentheses (ref. <steps>)) or none. THIS STATEMENT MUST BE IN EACH LINE. Try not to reference too many steps in each line, break up your argument.
-- There should be ONE answer line. This step should end in (final answer: <your answer>).
-- All non-answer steps should be referenced at least once (in a future (ref. <stems>)) to make your argument transparent and easy to follow.

DO NOT USE ANY OTHER FORMAT IN YOUR ANSWER. DO NOT ADD ADDITIONAL TEXT OUTSIDE OF THIS. Your output will be checked for errors and if found you will be asked to repeat (errors are marked below).

errors: {errors}

"""

PER_ANSWER_COT_INSTRUCTION = """Please answer each question thinking step-by-step. Argue in favor of the answer '{candidate_answer}' as convincingly as possible, even if you are not sure it is correct.

For example:

Question: Could the members of the Police perform lawful arrests?
Target answer: no
Answer (step-by-step):
1. My understanding is that the members of The Police were musicians. (ref. none)
2. Only law enforcement officers can perform lawful arrests (ref. none)
3. Based on 1 and 2, the answer is: no (ref. 1, 2) (final answer: no)

Another example:

Question: Is Newt Gingrich's nickname a type of Reptile?
Target answer: no
Answer (step-by-step):
1. Newt Gingrich was born Newton Leroy Gingrich and goes by the nickname Newt. (ref. none)
2. A newt is a type of Amphibian with smooth, sticky skin. (ref. none)
3. From what I recall, Amphibians are not reptiles. (ref. none)
4. Based on 1. 2 and 3, a newt is not a kind of reptile. (ref. 1, 2, 3)
5. Based on 4, the answer is 'no'. (ref. 4) (final answer: no)

Please only answer using this format where:

-- Every line of text must start with a number, including answer lines.
-- Every line at the end has a reference to past related facts (in parentheses (ref. <steps>)) or none. THIS STATEMENT MUST BE IN EACH LINE. Try not to reference too many steps in each line, break up your argument.
-- There should be ONE answer line. This step should end in (final answer: {candidate_answer}).
-- All non-answer steps should be referenced at least once (in a future (ref. <stems>)) to make your argument transparent and easy to follow.

DO NOT USE ANY OTHER FORMAT IN YOUR ANSWER. DO NOT ADD ADDITIONAL TEXT OUTSIDE OF THIS. Your output will be checked for errors and if found you will be asked to repeat (errors are marked below).

errors: {errors}

"""

PER_ANSWER_COT_INSTRUCTION_VERBAL = """Please answer each question thinking step-by-step. Argue in favor of the answer '{candidate_answer}' as convincingly as possible, even if you are not sure it is correct.  Include your confidence in each step to indicate your degree of belief (between 0 and 1) in that step.

BEFORE you write the numbered reasoning steps, output a single header line of the form

    (initial confidence: X%)

reporting how plausible '{candidate_answer}' is correct BEFORE you have argued for it. Use a wide range (5%-99%): low when the answer is clearly inconsistent with the question, high when it is the obvious answer.

For example:

Question: Could the members of the Police perform lawful arrests?
Target answer: no
Answer (step-by-step):
(initial confidence: 35%)
1. My understanding is that the members of The Police were musicians. (ref. none) (confidence: 95%)
2. Only law enforcement officers can perform lawful arrests (ref. none) (confidence: 90%)
3. Based on 1 and 2, the answer is: no (ref. 1, 2) (final answer: no) (confidence: 85%)

Another example:

Question: Is Newt Gingrich's nickname a type of Reptile?
Target answer: no
Answer (step-by-step):
(initial confidence: 80%)
1. Newt Gingrich was born Newton Leroy Gingrich and goes by the nickname Newt. (ref. none) (confidence: 95%)
2. A newt is a type of Amphibian with smooth, sticky skin. (ref. none) (confidence: 85%)
3. From what I recall, Amphibians are not reptiles. (ref. none) (confidence: 80%)
4. Based on 1. 2 and 3, a newt is not a kind of reptile. (ref. 1, 2, 3) (confidence: 80%)
5. Based on 4, the answer is 'no'. (ref. 4) (final answer: no) (confidence: 75%)

Please only answer using this format where:

-- The FIRST line is exactly '(initial confidence: X%)' — your prior on '{candidate_answer}' before any reasoning. NO step number, NO 'ref.' annotation.
-- Every numbered line of text must start with a number, including answer lines.
-- Every numbered line at the end has a reference to past related facts (in parentheses (ref. <steps>)) or none. THIS STATEMENT MUST BE IN EACH LINE. Try not to reference too many steps in each line, break up your argument.
-- Include (confidence: <X>%) on each numbered line indicating your confidence in that step.
-- There should be ONE answer line. This step should end in (final answer: {candidate_answer}).
-- All non-answer steps should be referenced at least once (in a future (ref. <stems>)) to make your argument transparent and easy to follow.

DO NOT USE ANY OTHER FORMAT IN YOUR ANSWER. DO NOT ADD ADDITIONAL TEXT OUTSIDE OF THIS. Your output will be checked for errors and if found you will be asked to repeat (errors are marked below).

errors: {errors}

"""

# ---------------------------------------------------------------------------
# Combined prompting strategy
# ---------------------------------------------------------------------------

COMBINED_COT_INSTRUCTION = """Please answer each question by arguing for BOTH sides step-by-step. First argue in favor of '{candidate_a}', then argue in favor of '{candidate_b}'. Include your confidence in each step to indicate your degree of belief (between 0 and 1) in that step, plus an initial confidence for each answer. 

Use section headers to separate the two arguments. Each section should have its own step numbering starting at 1.

For example:

Question: Could the members of the Police perform lawful arrests?
Answer (step-by-step for both sides):

## For yes (initial confidence: 65%):
1. The word 'Police' can refer to actual law enforcement officers. (ref. none) (confidence: 78%)
2. Law enforcement officers have the authority to perform lawful arrests. (ref. none) (confidence: 90%)
3. Based on 1 and 2, the answer is: yes (ref. 1, 2) (final answer: yes) (confidence: 65%)

## For no (initial confidence: 35%):
1. My understanding is that the members of The Police were musicians. (ref. none) (confidence: 89%)
2. Only law enforcement officers can perform lawful arrests. (ref. none) (confidence: 97%)
3. Based on 1 and 2, the answer is: no (ref. 1, 2) (final answer: no) (confidence: 85%)

Please only answer using this format where:

-- Each section starts with '## For <answer> (initial confidence: X%):' on its own line — the (initial confidence: X%) is your prior on that side BEFORE you have argued for it. The two sides' initial confidences should approximately sum to 100%.
-- Within each section, every line of text must start with a number, including answer lines.
-- Every line at the end has a reference to past related facts (in parentheses (ref. <steps>)) or none. THIS STATEMENT MUST BE IN EACH LINE. Try not to reference too many steps in each line, break up your argument.
-- Each section should have ONE answer line. This step should end in (final answer: <answer>).
-- Within each section, all non-answer steps should be referenced at least once (in a future (ref. <steps>)) to make your argument transparent and easy to follow.
-- Step numbers restart at 1 for each section. References are local to each section.

DO NOT USE ANY OTHER FORMAT IN YOUR ANSWER. DO NOT ADD ADDITIONAL TEXT OUTSIDE OF THIS. Your output will be checked for errors and if found you will be asked to repeat (errors are marked below).

errors: {errors}

"""

COMBINED_USER_TEMPLATE = """Question: {query}
Answer (step-by-step for both sides):
"""

# ============================================================================
# Multiple-choice (MCQ) prompts — for BBH and similar N-way tasks.
#
# Conventions kept identical to the binary prompts so the parser/runner
# treats both formats the same:
#   * (ref. <steps>) or (ref. none)   on every line
#   * (final answer: <letter>)        on the answer line
#   * (confidence: X%)                on every line in the *_VERBAL variants
#   * '## For <letter>:' section headers, each with its own numbered steps
#     starting at 1.
#
# Note: the COMBINED prompt asks the model to write a global
# '## Final Answer:' line; the per-answer prompts rely on each section's
# own (final answer: <letter>) token as the only commitment.
# ============================================================================

MCQ_USER_TEMPLATE = """Question: {query}
{choices_block}
Answer (step-by-step for all choices):
"""

MCQ_PER_ANSWER_USER_TEMPLATE = """Question: {query}
{choices_block}
Target answer: {candidate}
Answer (step-by-step):
"""

PER_ANSWER_COT_INSTRUCTION_MCQ_VERBAL = """Please answer the multiple-choice question step-by-step. Argue in favor of the answer '{candidate_answer}' as convincingly as possible, even if you are not sure it is correct. Include your confidence in each step as a percentage between 5 and 99 — be honest, low confidences are appropriate when an option is clearly inconsistent.

BEFORE you write the numbered reasoning steps, output a single header line of the form

    (initial confidence: X%)

reporting how plausible '{candidate_answer}' is correct BEFORE you have argued for it. Use a wide range (5%-99%): low when the option is clearly inconsistent with the question, high when it is the obvious answer.

For example:

Question: Three people - Alice, Bob, and Carol - finished a race. (1) Carol finished before Bob. (2) Alice did not finish first. Who finished first?
A. Alice
B. Bob
C. Carol
D. Cannot be determined
Target answer: C. Carol
Answer (step-by-step):
(initial confidence: 80%)
1. Constraint (2) says Alice did not finish first, so Alice is not first. (ref. none) (confidence: 99%)
2. Constraint (1) says Carol finished before Bob, so Bob is not first. (ref. none) (confidence: 99%)
3. The only remaining person is Carol. (ref. 1, 2) (confidence: 95%)
4. Based on 1, 2, 3, the answer is: C (ref. 1, 2, 3) (final answer: C) (confidence: 95%)

Please only answer using this format where:

-- The FIRST line is exactly '(initial confidence: X%)' — your prior on '{candidate_answer}' before any reasoning. NO step number, NO 'ref.' annotation.
-- Every numbered line starts with a number, including the answer line.
-- Every numbered line ends with (ref. <steps>) or (ref. none) AND (confidence: X%).
-- The confidence on each step reflects YOUR belief that the step is actually TRUE. Use a wide range (5%-99%).
-- There should be ONE answer line ending in (final answer: {candidate_letter}) (confidence: X%).
-- All non-answer steps should be referenced at least once.

DO NOT USE ANY OTHER FORMAT IN YOUR ANSWER.

errors: {errors}

"""

COMBINED_COT_INSTRUCTION_MCQ = """Please answer the multiple-choice question by building a SEPARATE step-by-step argument IN FAVOR of EACH of the {n} possible answers. In each choice's section, argue for THAT choice as convincingly as you can, as if you were its advocate, even if you believe it is wrong: build the strongest honest case that it is the correct answer. Do NOT evaluate, compare, or refute a choice inside its own section — a section must never conclude that its own choice is wrong or that a different choice is right. Save your overall judgment for the Final Answer section at the end.

Before each argument, give an initial confidence (between 5 and 99 as a percentage) reflecting how plausible that choice is BEFORE you have argued for it. Also give a confidence for each step: it is your belief that THAT STEP is true. A weak case for an implausible choice should show up as low-confidence steps (a step you doubt gets a low percentage), NOT as steps that argue against the choice.

Use section headers to separate the arguments. Each section starts at step 1.

For example:

Question: Three people - Alice, Bob, and Carol - finished a race. (1) Carol finished before Bob. (2) Alice did not finish first. Who finished first?
A. Alice
B. Bob
C. Carol
D. Cannot be determined

Answer (step-by-step for all choices):

## For A (initial confidence: 5%):
1. Alice is a listed finisher, so she is a candidate for first place. (ref. none) (confidence: 60%)
2. If constraint (2) were misread or applied to a different race, Alice finishing first would be consistent. (ref. 1) (confidence: 15%)
3. Based on 1, 2, the answer is: A (ref. 1, 2) (final answer: A) (confidence: 5%)

## For B (initial confidence: 5%):
1. Constraint (2) only excludes Alice, so on its own it leaves Bob as a possible first-place finisher. (ref. none) (confidence: 70%)
2. If constraint (1) does not apply to this race, Bob finishing first is compatible with the remaining information. (ref. 1) (confidence: 15%)
3. Based on 1, 2, the answer is: B (ref. 1, 2) (final answer: B) (confidence: 5%)

## For C (initial confidence: 80%):
1. Alice is excluded by constraint (2). (ref. none) (confidence: 99%)
2. Bob is excluded by constraint (1). (ref. none) (confidence: 98%)
3. Only Carol remains. (ref. 1, 2) (confidence: 95%)
4. Based on 1, 2, 3, the answer is: C (ref. 1, 2, 3) (final answer: C) (confidence: 95%)

## For D (initial confidence: 10%):
1. Only two constraints are given, which might not fully pin down the finishing order. (ref. none) (confidence: 40%)
2. If the constraints left more than one arrangement open, the answer would be undeterminable. (ref. 1) (confidence: 20%)
3. Based on 1, 2, the answer is: D (ref. 1, 2) (final answer: D) (confidence: 10%)

## Final Answer (initial confidence: 90%):
After considering all choices, the most consistent answer is: C (final answer: C)

Rules:
-- Each section starts with '## For <letter> (initial confidence: X%):' on its own line.
-- Every line within a section starts with a number and ends with (ref. <steps>) or (ref. none) AND (confidence: X%).
-- Different choices may need different numbers of reasoning steps — vary the length appropriately.
-- Every step in a section must SUPPORT that section's choice. Never write a step that argues against the section's own choice or that concludes another choice is correct.
-- The confidence reflects YOUR belief that the step is TRUE. Use a wide range (5%-99%). Express doubt about a weak choice through LOW step confidences, not through counter-arguments.
-- Each section has ONE answer line ending in (final answer: <letter>) (confidence: X%).
-- Step numbers restart at 1 for each section.
-- Provide arguments for ALL {n} choices in the order they appear.
-- After all sections, add a '## Final Answer (initial confidence: X%):' section with ONE line containing your final judgment and (final answer: <letter>). This is the ONLY place to compare choices.

DO NOT USE ANY OTHER FORMAT IN YOUR ANSWER.

errors: {errors}

"""

# ============================================================================
# Teacher-forcing prompt — parallel binary judgment (PURE answer prior)
# ============================================================================
# Used to derive slot 1 (the "logit" slot, .p) of each candidate's
# answer-prior Flip: P(option correct) = renormalised token logprob of
# 'correct' vs 'incorrect' at that option's position in the response.
# IMPORTANT: this prompt is INDEPENDENT of any CoT analysis. The model is
# asked to judge each option using only the question itself, so the
# resulting probabilities reflect the model's pure prior belief — not its
# post-rationalised conviction after being forced to argue for each side.
#
# Format of {choices_block} and {template_block} is built by the caller —
# uniform for both binary (yes/no) and MCQ (A/B/C/...).

TF_PARALLEL_TEMPLATE = """Question: {query}

Choices:
{choices_block}

You must judge EACH possible answer as either 'correct' or 'incorrect'. Do not skip any. Use this exact format (one line per option, one word after each em-dash, replacing each <correct or incorrect> placeholder):

{template_block}

Rules:
- Judge every option — output exactly {n} lines.
- Use only the words 'correct' or 'incorrect' (lowercase, no quotes, no punctuation).
- Exactly ONE option should be labelled 'correct'; all the others should be labelled 'incorrect'.
- Output only the {n} lines above and nothing else.
"""
