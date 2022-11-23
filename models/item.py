import random
from typing import List
from dataclasses import dataclass

import randomname


@dataclass
class Item:
    name: str
    value: float


@dataclass
class ItemResponse:
    items: List[Item]


def generate_random_items(num_items: int = 10) -> List[Item]:
    return [Item(randomname.get_name(), random.random()) for i in range(num_items)]
