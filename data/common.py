import re
from typing import List
from collections import OrderedDict

def tokens_alignment(seq: List, sub_seq: List, ) -> List[List[int]]:
    """
    Sequence alignment by Needleman-Wunsch algorithm.

    If subsequence element is not in sequence, then return -100
    video:  https://youtu.be/ipp-pNRIp4g

    Returns:
        List of [sub_seq index, seq index]
    """
    # we trace back from the bottom right corner
    seq = seq[::-1]
    sub_seq = sub_seq[::-1]

    # if x == y, add 1 to score, else add -1
    match_score = [-1, 1]
    gap = -2
    col = len(seq)
    row = len(sub_seq)

    dp_table = [[0]*(col+1) for _ in range(row+1)]

    result = [[i, -100] for i in range(0, row)]

    for c in range(0, col+1):
        dp_table[0][c] = gap * c

    for r in range(0, row+1):
        dp_table[r][0] = gap * r

    for r in range(1, row+1):
        for c in range(1, col+1):
            val_left = dp_table[r][c-1] + gap
            val_up = dp_table[r-1][c] + gap
            val_left_up = dp_table[r-1][c-1] + \
                match_score[sub_seq[r-1] == seq[c-1]]
            dp_table[r][c] = max(val_left, val_up, val_left_up)

    # traceback
    r, c = row, col

    while r > 0 and c > 0:
        if sub_seq[r-1] == seq[c-1]:  # match, go left-up
            result[r-1][1] = c-1
            r -= 1
            c -= 1
        else:  # not match, go to highest neighbor
            val_left = dp_table[r][c-1]
            val_up = dp_table[r-1][c]
            val_left_up = dp_table[r-1][c-1]

            # if value is same, the order is left-top > up > left
            if val_left_up >= val_left and val_left_up >= val_up:   # left-up
                r -= 1
                c -= 1
            elif val_up >= val_left and val_up >= val_left_up:  # up
                r -= 1
            else:
                c -= 1

    for i in range(len(result)):
        # row = len(sub_seq)
        result[i][0] = row - result[i][0] - 1

        if result[i][1] != -100:
            # col = len(seq)
            result[i][1] = col - result[i][1] - 1

    result = result[::-1]

    return result