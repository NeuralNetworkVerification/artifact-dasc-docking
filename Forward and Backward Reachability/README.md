Forward and Backward Reachability code files for reference. We outline files and a brief description of included functions. We leave such files as documentation and reference for future end-to-end pipelines/case studies.

mod_verification.py: File defining functions for backward reachability. Running the file runs a backward reachability analysis from starting region defined in line 153.
myCell.py: File containing class and helper functions for mod_verification.py.
demo_bounds.py: File running a forward reachability script. The number of steps for forward reachability is defined on line 12 (NUM_STEPS variable) and initial input bounds for state components are defined on lines 126-134. 