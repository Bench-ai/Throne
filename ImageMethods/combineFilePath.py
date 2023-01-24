import os

def combine_method(curr, add, before):
    if before:
        combined = os.path.join(add, curr)
    else:
        combined = os.path.join(curr, add)

    if os.path.exists(combined):
        return combined
    else:
        return "Path is invalid"
    return combined

