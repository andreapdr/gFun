from tqdm import tqdm
import re
import sys


def mask_numbers(data, number_mask='numbermask'):
    mask = re.compile(r'\b[0-9][0-9.,-]*\b')
    masked = []
    for text in tqdm(data, desc='masking numbers'):
        masked.append(mask.sub(number_mask, text))
    return masked