Files:
train_data.csv → Training dataset used to create the training and validation splits.
test_data.csv → Test dataset used only for the final evaluation.

CSV files with columns:
Sequence: amino-acid sequence.
label: binary aggregation label.

Binary label meaning. This follows the binary prediction framing used in peptide aggregation benchmarks such as AggBERT: https://pmc.ncbi.nlm.nih.gov/articles/PMC10777593/
Label = 1 means the peptide is treated as aggregation-positive / amyloidogenic; 
Label = 0 means aggregation-negative / non-amyloidogenic. 
