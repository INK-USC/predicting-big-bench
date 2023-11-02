from .basic import preprocess as basic_preprocess
from .multitarget import preprocess as multitarget_preprocess
from .cf import preprocess as cf_preprocess
from .no_model_name import preprocess as no_model_name_preprocess

def load_data(args, logger, mode="basic"):
    load_funcs = {
        "basic": basic_preprocess,
        "multitarget": multitarget_preprocess,
        "cf": cf_preprocess,
        "no_model_name": no_model_name_preprocess,
    }
    load_func = load_funcs[mode]
    return load_func(args, logger, args.train_file, args.dev_file, args.test_file)
