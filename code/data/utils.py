import os

def normalize_score(row):
    """normalize score to the range of 0-1"""
    metric = row['metric_name']
    score = row['score']
    score_min, score_max = SCORE_RANGES[metric]
    return (score - score_min) / (score_max - score_min)


# ref: https://github.com/google/BIG-bench/blob/main/bigbench/api/json_task.py
SCORE_RANGES = {
    "bleu": [0.0, 100.0],
    "sequence_f1": [0.0, 100.0],
    "bleurt": [
        -0.1,
        1.1,
    ],  # BLEURT-20 is roughly calibrated to [0, 1]; https://github.com/google-research/bleurt#interpreting-bleurt-scores
    "calibration_multiple_choice_brier_score": [0.0, 1.0],
    "expected_calibration_error": [0.0, 1.0],
    "exact_str_match": [0.0, 1.0],
    "case_insensitive_str_match": [0.0, 1.0],
    "log_likelihood": [-10, 0.0],
    "multiple_choice_grade": [0.0, 1.0],
    "numeric_match_with_0_1_relative_error": [0.0, 1.0],
    "rouge1": [0.0, 100.0],
    "rouge2": [0.0, 100.0],
    "rougeLsum": [0.0, 100.0], ## for convenience this is taken care of elsewhere...
    "weighted_log_probabilities": [-10, 0.0],
    "weighted_probabilities": [-1, 1],
    "macro_f1": [0.0, 1.0],
    # added in this work...
    "log_likelihood_per_char": [-10, 0.0],
    "log_likelihood_per_word": [-10, 0.0],
    "normalized_aggregate_score": [-10, 100]
}

# removing log_likelihood-style metrics because the ranges are very different
METRICS_OF_INTEREST = [
    "exact_str_match",
    "multiple_choice_grade", 
    "rougeLsum", 
    "bleu", 
    "sequence_f1", 
    "bleurt", 
    "case_insensitive_str_match",
    "numeric_match_with_0_1_relative_error",
    "rouge1",
    "rouge2",
    "macro_f1"
]

BBLITE_TASKS = ['bbq_lite_json:bbq_lite_json_age_ambig', 'bbq_lite_json:bbq_lite_json_age_disambig', 'bbq_lite_json:bbq_lite_json_disability_status_ambig', 'bbq_lite_json:bbq_lite_json_disability_status_disambig', 'bbq_lite_json:bbq_lite_json_gender_identity_ambig', 'bbq_lite_json:bbq_lite_json_gender_identity_disambig', 'bbq_lite_json:bbq_lite_json_nationality_ambig', 'bbq_lite_json:bbq_lite_json_nationality_disambig', 'bbq_lite_json:bbq_lite_json_physical_appearance_ambig', 'bbq_lite_json:bbq_lite_json_physical_appearance_disambig', 'bbq_lite_json:bbq_lite_json_race_ethnicity_ambig', 'bbq_lite_json:bbq_lite_json_race_ethnicity_disambig', 'bbq_lite_json:bbq_lite_json_religion_ambig', 'bbq_lite_json:bbq_lite_json_religion_disambig', 'bbq_lite_json:bbq_lite_json_ses_ambig', 'bbq_lite_json:bbq_lite_json_ses_disambig', 'bbq_lite_json:bbq_lite_json_sexual_orientation_ambig', 'bbq_lite_json:bbq_lite_json_sexual_orientation_disambig', 'code_line_description', 'conceptual_combinations:emergent_properties', 'formal_fallacies_syllogisms_negation', 'hindu_knowledge', 'language_identification', 'linguistics_puzzles', 'logic_grid_puzzle', 'logical_deduction:five_objects', 'logical_deduction:seven_objects', 'logical_deduction:three_objects', 'novel_concepts', 'operators', 'parsinlu_reading_comprehension', 'play_dialog_same_or_different', 'strange_stories:boolean', 'strange_stories:multiple_choice', 'strategyqa', 'symbol_interpretation:adversarial', 'symbol_interpretation:emoji_agnostic', 'symbol_interpretation:name_agnostic', 'symbol_interpretation:plain', 'symbol_interpretation:tricky', 'vitaminc_fact_verification', 'winowhy']
BBHARD_SUBTASKS = ['causal_judgment', 'date_understanding', 'disambiguation_qa', 'dyck_languages', 'formal_fallacies_syllogisms_negation', 'geometric_shapes', 'hyperbaton', 'logical_deduction:five_objects', 'logical_deduction:seven_objects', 'logical_deduction:three_objects', 'movie_recommendation', 'navigate', 'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding', 'temporal_sequences', 'tracking_shuffled_objects:five_objects', 'tracking_shuffled_objects:seven_objects', 'tracking_shuffled_objects:three_objects', 'word_sorting']