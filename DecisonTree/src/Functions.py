

def gini_index(groups, labels):

    n_instances = float(sum([len(group) for group in groups]))

    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0

        for class_val in labels:
            proportion = group.count(class_val) / size
            score += proportion * proportion
            
        gini += (1.0 - score) * (size / n_instances)

    return gini


def entropy(groups, labels):
    from math import log2
    n_instances = float(sum([len(group) for group in groups]))
    entropy_value = 0.0
    
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in labels:
            proportion = group.count(class_val) / size
            if proportion > 0:
                score -= proportion * log2(proportion)
        entropy_value += (score * (size / n_instances))
    
    return entropy_value