# id  x   y   z psi
# 0    

from collections import OrderedDict
import numpy as np


april_tag_dict = OrderedDict()

april_tag_dict[0] = {'id': int(0), 'position': np.ndarray(0, -4.4), 'depth': -12.3,  'heading': 90.0}
april_tag_dict[1] = {'id': int(1), 'position': np.ndarray(0, 4.4), 'depth': -12.3,  'heading': -90.0}
april_tag_dict[2] = {'id': int(2), 'position': np.ndarray(0, -4.4), 'depth': -16.0,  'heading': 90.0}
april_tag_dict[3] = {'id': int(3), 'position': np.ndarray(0, -4.4), 'depth': -17.3,  'heading': 90.0}
april_tag_dict[4] = {'id': int(2), 'position': np.ndarray(0, 4.4), 'depth': -16.0,  'heading': -90.0}
april_tag_dict[5] = {'id': int(5), 'position': np.ndarray(0, 4.4), 'depth': -17.3,  'heading': -90.0}
