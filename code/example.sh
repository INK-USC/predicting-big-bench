setting="l1" # select from "l1", "l2_1", "l2_2", "l3", "l4"
model_arch="mlp" # select from "bsl_model_task", "svd", "task_task_knn", "model_model_knn","random_forest", "xgb", "mlp"
n_trials=200 # random hyperparameter combinations to try

python cli.py \
--train_file ../data/bigbench/${setting}/0/train.csv \
--dev_file ../data/bigbench/${setting}/0/dev.csv \
--test_file ../data/bigbench/${setting}/0/test.csv \
--data_dir ../data/bigbench/${setting}/ \
--mode tunehp_then_multi_run \
--model_arch ${model_arch} \
--output_dir output/${setting}/${model_arch} \
--preferred \
--save_predictions \
--n_trials ${n_trials}