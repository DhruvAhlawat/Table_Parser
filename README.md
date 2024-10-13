A *Table query answering software*, based only on **LSTMs**, and uses attention for increased accuracy.
The training and evaluation framework is **highly optimized**, and it can train or evaluate on tables with variable number of columns and rows, working even on tables with **> 30000 rows**, where Language model based approaches fail due to context length.

further improvements include adding multiple attention heads with the LSTM formation. Can answer with multiple cells being the answer, as long as the cells are in the same column, although not with very high accuracy.

It was made for completing an assignment in IIT Delhi's NLP course, where I gained distinction and a full extra point (among 4 others) for this submission.

training is done in the following way:
```
    bash run_model.sh <train_jsonfile> <val_jsonfile>
```
and inference in the following way:
```
    bash run_model.sh <test_jsonfile> <output_name_file>
```

where the json files are in the format of the json files in the data folder. look at $ final_evaluation.sh $ for an example.


