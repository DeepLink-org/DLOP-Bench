export PYTHONPATH=./long_tail_bench:$PYTHONPATH
# FRAMEWORK=torch SAMPLE_IMPL=torch python ./long_tail_bench/api/api.py -st 1 -c $1 masked_fill
# FRAMEWORK=torch SAMPLE_IMPL=torch python ./long_tail_bench/api/api.py -st 1 -c $1 index_select
# FRAMEWORK=torch SAMPLE_IMPL=torch python ./long_tail_bench/api/api.py -st 1 -c $1 masked_fill_
FRAMEWORK=torch SAMPLE_IMPL=torch python ./long_tail_bench/api/api.py -st 1 -c $1 cat
# FRAMEWORK=torch SAMPLE_IMPL=torch python ./long_tail_bench/api/api.py -st 1 -c $1 stack
# FRAMEWORK=torch SAMPLE_IMPL=torch python ./long_tail_bench/api/api.py -st 1 -c $1 transpose
# FRAMEWORK=torch SAMPLE_IMPL=torch python ./long_tail_bench/api/api.py -st 1 -c $1 permute
# FRAMEWORK=torch SAMPLE_IMPL=torch python ./long_tail_bench/api/api.py -st 1 -c $1 squeeze
# FRAMEWORK=torch SAMPLE_IMPL=torch python ./long_tail_bench/api/api.py -st 1 -c $1 view
# FRAMEWORK=torch SAMPLE_IMPL=torch python ./long_tail_bench/api/api.py -st 1 -c $1 reshape
# FRAMEWORK=torch SAMPLE_IMPL=torch python ./long_tail_bench/api/api.py -st 1 -c $1 unsqueeze
# FRAMEWORK=torch SAMPLE_IMPL=torch python ./long_tail_bench/api/api.py -st 1 -c $1 unbind
# FRAMEWORK=torch SAMPLE_IMPL=torch python ./long_tail_bench/api/api.py -st 1 -c $1 nonzero
# FRAMEWORK=torch SAMPLE_IMPL=torch python ./long_tail_bench/api/api.py -st 1 -c $1 unique
# FRAMEWORK=torch SAMPLE_IMPL=torch python ./long_tail_bench/api/api.py -st 1 -c $1 where
# FRAMEWORK=torch SAMPLE_IMPL=torch python ./long_tail_bench/api/api.py -st 1 -c $1 histc