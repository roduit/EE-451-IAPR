# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-05-03 -*-
# -*- Last revision: 2024-05-03 -*-
# -*- python version : 3.9.18 -*-
# -*- Description: Class to load data -*-

# Importing files
from data_classes.data import Coin

class refCoin(Coin):
    """
    Class to load the reference data
    """
    def __init__(self):
        super().__init__('ref')
        self.load_data()