import os
import numpy as np

def combine_method(curr, add, before):
    combined = os.path.join(add, curr) if before else os.path.join(curr, add)

    return combined if os.path.exists(combined) else np.NaN
