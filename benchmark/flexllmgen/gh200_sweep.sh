#! /bin/bash

python bench_suite.py 30b_allcpu --log-file 30b_allcpu.log
python bench_suite.py 30b_2575 --log-file 30b_2575.log
python bench_suite.py 30b_5050 --log-file 30b_5050.log
python bench_suite.py 30b_7525 --log-file 30b_7525.log
python bench_suite.py 30b_allgpu --log-file 30b_allgpu.log
