#!/bin/bash

python experiments/causal_trace.py --fact_file data/MedFE_casual_tracing.json

python experiments/causal_trace.py --fact_file data/MedCF_casual_tracing.json