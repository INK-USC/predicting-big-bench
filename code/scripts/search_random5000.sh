cd ..

NTASK=42
search_mode="random" # ["fixed", "random", "greedy", "beam", "factored_beam", "ga", "sa"]

python cli.py \
--full_file ../data/bigbench/filtered_v2.csv \
--hp "{'lr': 0.001, 'batch_size': 128, 'dropout': 0.0, 'hidden_dims': (128, 64, 32, 16), 'weight_decay': 1e-05}" \
--mode search \
--model_arch mlp \
--search_mode random \
--output_dir output/search/random5000/ntask${NTASK} \
--search_n_trials 5000 \
--search_cv_level l0 \
--search_budget ${NTASK} \
--save_search_logs \
--search_subroutine mlp_for_search \
--preferred;