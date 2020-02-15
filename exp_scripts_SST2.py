from models.word_SST2 import train
from models.word_SST2 import test_model as SST2_test_model
from models.word_SST2 import adv_train as SST2_adv_train
from diffai.exhaustive import SwapSub, DelDupSubWord, DelDupSubChar
from functools import partial
import os


SST2_train("word_SST2")

target_transformations = ["Composition(delete, delete, sub, sub)", "Composition(ins, ins, sub, sub)", "Composition(delete, delete, ins, ins, sub, sub)"]
funcs = [partial(DelDupSubWord, 2, 0, 2), partial(DelDupSubWord, 0, 2, 2), partial(DelDupSubWord, 2, 2, 2)]
model_names = ["delsub", "inssub", "delinssub"]
for (target_transformation, func, model_name) in zip(target_transformations, funcs, model_names):
    SST2_adv_train("word_SST2_%s_aug" % model_name, target_transformation=target_transformation, adv_train_random=True)
    SST2_adv_train("word_SST2_%s_adv" % model_name, target_transformation=target_transformation, adv_train_random=False)
    models = ["word_SST2", "word_SST2_%s_aug" % model_name, "word_SST2_%s_adv" % model_name]
    for model in models:
        SST2_test_model(model, target_transformation=target_transformation)
        SST2_test_model(model, func=func)
