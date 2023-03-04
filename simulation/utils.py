def merge_sorted(list_a, list_b, using):
    if len(list_a) == 0:
        return list_b
    if len(list_b) == 0:
        return list_a
    i = 0
    j = 0
    res = []
    while i < len(list_a) and j < len(list_b):
        if using(list_a[i]) < using(list_b[j]):
            res.append(list_a[i])
            i += 1
        else:
            res.append(list_b[j])
            j += 1
    return res + list_a[i:] + list_b[j:]

