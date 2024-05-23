# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-05-03 -*-
# -*- Last revision: 2024-05-03 -*-
# -*- python version : 3.12.3 -*-
# -*- Description: Class to load ref data -*-

# Importing files
from data_classes.data import Coin

class refCoin(Coin):
    """
    Class to load the reference data
    """
    def __init__(self, save=False):

        super().__init__(type='ref', save=save)
        
        self.load_data()