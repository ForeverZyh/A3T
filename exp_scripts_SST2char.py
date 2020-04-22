from models.char_SST2 import train as SST2_train
from models.char_SST2 import test_model as SST2_test_modqel
from models.char_SST2 import adv_train as SST2_adv_train
from diffai.exhaustive import SwapSub, DelDupSubWord, DelDupSubChar
from functools import partial
import os


SST2_train("char_SST2")

target_transformations = ["Composition(swap, sub)", "Composition(delete, sub)", "Composition(ins, sub)"]
funcs = [partial(SwapSub, 1, 1), partial(DelDupSubChar, 1, 0, 1), partial(DelDupSubChar, 0, 1, 1)]
model_names = ["swapsub", "delsub", "inssub"]
for (target_transformation, func, model_name) in zip(target_transformations, funcs, model_names):
    SST2_adv_train("char_SST2_%s_aug" % model_name, target_transformation=target_transformation, adv_train_random=True)
    SST2_adv_train("char_SST2_%s_adv" % model_name, target_transformation=target_transformation, adv_train_random=False)
    models = ["char_SST2", "char_SST2_%s_aug" % model_name, "char_SST2_%s_adv" % model_name]
    for model in models:
        SST2_test_model(model, target_transformation=target_transformation)
        SST2_test_model(model, func=func)
