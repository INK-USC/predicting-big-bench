cd ..

python cli.py \
--full_file ../data/bigbench/filtered_v2.csv \
--hp "{'lr': 0.001, 'batch_size': 128, 'dropout': 0.0, 'hidden_dims': (128, 64, 32, 16), 'weight_decay': 1e-05}" \
--mode search \
--model_arch mlp \
--search_mode beam \
--output_dir output/search/greedy \
--search_cv_level l0 \
--save_search_logs \
--search_beam_size 1 \
--search_beam_random 1.0 \
--search_budget 42 \
--search_subroutine mlp_for_search \
--preferred;