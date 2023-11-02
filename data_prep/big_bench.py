"""
Parse Big-Bench results into a csv file
"""
import os
import re
import json
import tqdm
import pandas as pd

BASE_DIR = "../../BIG-bench/bigbench/benchmark_tasks"

# copied from https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/keywords_to_tasks.md#json
JSON_TASKS = "abstract_narrative_understanding, anachronisms, analogical_similarity, analytic_entailment, arithmetic, ascii_word_recognition, authorship_verification, auto_categorization, auto_debugging, bbq_lite_json, bridging_anaphora_resolution_barqa, causal_judgment, cause_and_effect, checkmate_in_one, chess_state_tracking, chinese_remainder_theorem, cifar10_classification, code_line_description, codenames, color, common_morpheme, conceptual_combinations, conlang_translation, contextual_parametric_knowledge_conflicts, crash_blossom, crass_ai, cryobiology_spanish, cryptonite, cs_algorithms, dark_humor_detection, date_understanding, disambiguation_qa, discourse_marker_prediction, disfl_qa, dyck_languages, elementary_math_qa, emoji_movie, emojis_emotion_prediction, empirical_judgments, english_proverbs, english_russian_proverbs, entailed_polarity, entailed_polarity_hindi, epistemic_reasoning, evaluating_information_essentiality, fact_checker, fantasy_reasoning, few_shot_nlg, figure_of_speech_detection, formal_fallacies_syllogisms_negation, gem, gender_inclusive_sentences_german, general_knowledge, geometric_shapes, goal_step_wikihow, gre_reading_comprehension, hhh_alignment, hindi_question_answering, hindu_knowledge, hinglish_toxicity, human_organs_senses, hyperbaton, identify_math_theorems, identify_odd_metaphor, implicatures, implicit_relations, indic_cause_and_effect, intent_recognition, international_phonetic_alphabet_nli, international_phonetic_alphabet_transliterate, intersect_geometry, irony_identification, kanji_ascii, kannada, key_value_maps, known_unknowns, language_games, language_identification, linguistic_mappings, linguistics_puzzles, list_functions, logic_grid_puzzle, logical_args, logical_deduction, logical_fallacy_detection, logical_sequence, mathematical_induction, matrixshapes, medical_questions_russian, metaphor_boolean, metaphor_understanding, minute_mysteries_qa, misconceptions, misconceptions_russian, mnist_ascii, modified_arithmetic, moral_permissibility, movie_dialog_same_or_different, movie_recommendation, mult_data_wrangling, multiemo, natural_instructions, navigate, nonsense_words_grammar, novel_concepts, object_counting, odd_one_out, operators, paragraph_segmentation, parsinlu_qa, parsinlu_reading_comprehension, penguins_in_a_table, periodic_elements, persian_idioms, phrase_relatedness, physical_intuition, physics, physics_questions, play_dialog_same_or_different, polish_sequence_labeling, presuppositions_as_nli, qa_wikidata, question_selection, real_or_fake_text, reasoning_about_colored_objects, repeat_copy_logic, rephrase, rhyming, riddle_sense, ruin_names, salient_translation_error_detection, scientific_press_release, semantic_parsing_in_context_sparc, semantic_parsing_spider, sentence_ambiguity, similarities_abstraction, simp_turing_concept, simple_arithmetic_json, simple_arithmetic_json_multiple_choice, simple_arithmetic_json_subtasks, simple_arithmetic_multiple_targets_json, simple_ethical_questions, simple_text_editing, snarks, social_iqa, social_support, sports_understanding, strange_stories, strategyqa, sufficient_information, suicide_risk, swahili_english_proverbs, swedish_to_german_proverbs, symbol_interpretation, tellmewhy, temporal_sequences, tense, timedial, topical_chat, tracking_shuffled_objects, understanding_fables, undo_permutation, unit_conversion, unit_interpretation, unnatural_in_context_learning, vitaminc_fact_verification, what_is_the_tao, which_wiki_edit, wino_x_german, winowhy, word_sorting, word_unscrambling".split(", ")

def main():
    all_task_names = sorted(filter(lambda x: os.path.isdir(os.path.join(BASE_DIR, x)) and x != "results", os.listdir(BASE_DIR)))
    for task in JSON_TASKS:
        assert task in all_task_names

    df = pd.DataFrame(columns=["task", "subtask", "model_family", "model_name", "non_embedding_params", "flop_matched_non_embedding_params", "total_params", "n_shot", "metric_name", "is_preferred_metric", "score"])

    for task_name in tqdm.tqdm(JSON_TASKS):

        if task_name in ["list_functions", "mult_data_wrangling", "multiemo"]:
            continue

        print(task_name)

        result_dir = os.path.join(BASE_DIR, task_name, "results")
        result_files = sorted(filter(lambda x: x.endswith(".json"), os.listdir(result_dir)))
        # print(result_files)

        for result_file in result_files:
            with open(os.path.join(BASE_DIR, task_name, "results", result_file)) as fin:
                json_data = json.load(fin)

            for entry in json_data["scores"]:

                # skip entries that are aggregating subtask performance
                if entry["preferred_score"] == "normalized_aggregate_score":
                    continue

                all_metrics = entry["score_dict"].keys()

                for metric in all_metrics:

                    score = entry["score_dict"][metric]


                    df.loc[len(df.index)] = [
                        task_name,
                        entry["subtask_description"],
                        json_data["model"]["model_family"],
                        json_data["model"]["model_name"],
                        json_data["model"]["non_embedding_params"],
                        json_data["model"]["flop_matched_non_embedding_params"],
                        json_data["model"]["total_params"],
                        entry["number_of_shots"],
                        metric,
                        int(metric == entry["preferred_score"]),
                        score
                    ]


    print(df.head())

    df.to_csv("../data/bigbench/all.csv", index=False)

if __name__ == "__main__":
    main()
