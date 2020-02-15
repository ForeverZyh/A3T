from models.char_AG import train as AG_train
from models.char_AG import test_model as AG_test_modqel
from models.char_AG import adv_train as AG_adv_train
from diffai.exhaustive import SwapSub, DelDupSubWord, DelDupSubChar
from functools import partial
import os


AG_train("char_AG")

target_transformations = ["Composition(swap, sub)", "Composition(delete, sub)", "Composition(ins, sub)"]
funcs = [partial(SwapSub, 1, 1), partial(DelDupSubChar, 1, 0, 1), partial(DelDupSubChar, 0, 1, 1)]
model_names = ["swapsub", "delsub", "inssub"]
for (target_transformation, func, model_name) in zip(target_transformations, funcs, model_names):
    AG_adv_train("char_AG_%s_aug" % model_name, target_transformation=target_transformation, adv_train_random=True)
    AG_adv_train("char_AG_%s_adv" % model_name, target_transformation=target_transformation, adv_train_random=False)
    models = ["char_AG", "char_AG_%s_aug" % model_name, "char_AG_%s_adv" % model_name]
    for model in models:
        AG_test_model(model, target_transformation=target_transformation)
        AG_test_model(model, func=func)
