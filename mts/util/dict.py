from typing import Dict, Any, Union, List, Tuple, Sequence

KwargsType = Dict[str, Any]


def extract_kwargs(
    kwargs: KwargsType,
    what_dict: Union[List[str], str],
    merge: bool = False,
    sep: str = "::",
) -> Tuple[Dict[str, KwargsType], KwargsType]:
    if not isinstance(what_dict, list):
        what_dict = [what_dict]

    what_dict = {w: {} for w in what_dict}
    left_dict = {}
    for k, v in kwargs.items():
        try:
            group, name = k.split(sep, 1)
        except ValueError:
            if merge:
                for wv in what_dict.values():
                    wv[k] = v
            else:
                left_dict[k] = v
        else:
            group_dict = what_dict.get(group)
            if group_dict is not None:
                group_dict[name] = v
            else:
                if merge:
                    for wv in what_dict.values():
                        wv[k] = v
                else:
                    left_dict[k] = v
    if merge:
        return what_dict
    return what_dict, left_dict


def dict_from_dups(dups: Sequence[Tuple[Any, Any]]) -> Dict[Any, List[Any]]:
    dictt = {}
    for k, v in dups:
        dictt.setdefault(k, []).append(v)
    return dictt
