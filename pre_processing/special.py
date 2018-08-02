WEIGHT_AMP = 0.85


def two_type(result):
    new_result = [0, 0]
    if result.index(1) == 0:
        new_result[0] = 1
    else:
        new_result[1] = 1
    return new_result


def three_type(result):
    new_result = [0, 0, 0]
    if result.index(1) == 0:
        new_result[0] = 1
    elif result.index(1) == 1:
        new_result[1] = 1
    else:
        new_result[2] = 1
    return new_result


def two_type_weight(count):
    base_weight_0 = (count[0] + count[1]) / count[0]
    base_weight_1 = (count[0] + count[1]) / count[1]
    weight_0 = base_weight_0 * WEIGHT_AMP
    weight_1 = int(base_weight_1)
    return {0: weight_0, 1: weight_1}


def three_type_weight(count):
    base_weight_0 = (count[0] + count[1] + count[2]) / count[0]
    base_weight_1 = (count[0] + count[1] + count[2]) / count[1]
    base_weight_2 = (count[0] + count[1] + count[2]) / count[2]
    return {0: base_weight_0, 1: base_weight_1, 2: base_weight_2}
