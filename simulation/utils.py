from typing import Tuple, TypeVar, Union, List, Callable, Iterable


T = TypeVar('T')
X = TypeVar('X')


def binary_search(ls: List[T], diff_func: Callable[[T], X],
                  return_index: bool = False,
                  preference: str = 'nearest') -> Union[Tuple[T, int], T, None]:
    """
    Binary search via predicate.
    preference param:
    - 'nearest': with smallest diff, result always exists in non-empty list
    - 'next': strictly larger
    - 'prev': strictly smaller
    """
    if preference not in ('nearest', 'next', 'prev'):
        raise ValueError('binary search: invalid preference: ' + preference)

    if len(ls) == 0:
        return None

    l = 0
    r = len(ls)
    while l < r:
        m = l + (r - l) // 2
        cmp_res = diff_func(ls[m])
        if cmp_res == 0:
            return (ls[m], m) if return_index else ls[m]
        elif cmp_res < 0:
            r = m
        else:
            l = m + 1

    if l >= len(ls):
        l -= 1

    if (preference == 'nearest') and (l > 0) and (abs(diff_func(ls[l - 1])) < abs(diff_func(ls[l]))):
        l -= 1
    elif (preference == 'prev') and (diff_func(ls[l]) < 0):
        if l > 0:
            l -= 1
        else:
            return None
    elif (preference == 'next') and (diff_func(ls[l]) > 0):
        if l < len(ls) - 1:
            l += 1
        else:
            return None

    return (ls[l], l) if return_index else ls[l]


def differs_from(x: T, using=None) -> Callable[[T], X]:
    def _diff(y):
        if using is not None:
            y = using(y)
        return x - y

    return _diff


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

