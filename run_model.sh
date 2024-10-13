# !/bin/bash

if [ "$1" = "test" ]; then
    conda run final_test.py $2 $3 
else
    conda run final_train.py $1 $2
fi