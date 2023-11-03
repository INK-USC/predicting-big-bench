cd ..

python cli.py \
--full_file ../data/bigbench/filtered_v2.csv \
--hp "{'lr': 0.001, 'batch_size': 128, 'dropout': 0.0, 'hidden_dims': (128, 64, 32, 16), 'weight_decay': 1e-05}" \
--mode search \
--model_arch mlp \
--output_dir output/search_eval/bbhard \
--search_cv_level l2 \
--selected_tasks bbhard \
--preferred;

python cli.py \
--full_file ../data/bigbench/filtered_v2.csv \
--hp "{'lr': 0.001, 'batch_size': 128, 'dropout': 0.0, 'hidden_dims': (128, 64, 32, 16), 'weight_decay': 1e-05}" \
--mode search \
--model_arch mlp \
--output_dir output/search_eval/bblite \
--search_cv_level l2 \
--selected_tasks bblite \
--preferred;