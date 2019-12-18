from models.char_AG import train, adv_train, test_model
from DSL.unit_tests import do_test

if __name__ == "__main__":
    # do_test()
    # train("char_AG")
    # adv_train("char_AG", "char_AG_adv")
    test_model("char_AG")