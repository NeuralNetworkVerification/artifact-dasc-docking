Grid reachability code files for reference. We outline files and a brief description of included functions. We leave such files as documentation and reference for future end-to-end pipelines/case studies.


queries.py: Functions used in search_procedure.py using Marabou for verification, including construction of edges between grid cells. 

search_procedure.py: Contains various functions for defining a grid of cells, determining their associated bounds such that adjacent cells only must be checked (as stated in the associated paper), functions to determine transitions/edges between cells, and to determine cells which reach the docking region. Running this file produces a set of cells over defined bounds.

testing.py: Contains functions for analysis of cycles and whether states are positive or negative given cells and edge relationships. Running this file given a set of file paths outputted from a SLURM cluster set of jobs (in which functions present in search_procedure.py is run) run conducts one such analysis.

parse_text_file_to_csv.py: Defines functions to convert text files produced by functions in search_procedure.py to csv files, as well as conduct an analysis of these csv files with respect to cardinality of state components.