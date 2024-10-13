export CUDA_VISIBLE_DEVICES=0
SHOULD_TRAIN=$1

# Must be run from within the directory containing the run_model.sh script

if [ $SHOULD_TRAIN == "true" ]; then
    echo "Starting Training"
    timeout "24h" bash run_model.sh data/A2_train.jsonl ../data/A2_val.jsonl #trains for a day at max.
fi

echo "Starting Evaluation"
bash run_model.sh test data/A2_test.jsonl predictions_new.jsonl


echo "Evaluating Submission"
python evaluate_submission.py data/A2_test.jsonl predictions_new.jsonl