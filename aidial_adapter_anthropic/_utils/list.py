from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Container,
    Generic,
    Iterable,
    List,
    Self,
    Set,
    Tuple,
    TypeVar,
)

_T = TypeVar("_T")
_V = TypeVar("_V")


def select_by_indices(lst: List[_T], indices: Container[int]) -> List[_T]:
    return [elem for idx, elem in enumerate(lst) if idx in indices]


def omit_by_indices(lst: List[_T], indices: Container[int]) -> List[_T]:
    return [elem for idx, elem in enumerate(lst) if idx not in indices]


def group_by(
    lst: List[_T],
    key: Callable[[_T], Any],
    init: Callable[[_T], _V],
    merge: Callable[[_V, _T], _V],
) -> List[_V]:

    def _gen():
        if not lst:
            return

        prev_val = init(lst[0])
        prev_key = key(lst[0])

        for elem in lst[1:]:
            if prev_key == key(elem):
                prev_val = merge(prev_val, elem)
            else:
                yield prev_val
                prev_val = init(elem)
                prev_key = key(elem)

        yield prev_val

    return list(_gen())


@dataclass
class ListProjection(Generic[_T]):
    """
    The class represents a transformation of the original list which may
    include merge, removal and addition of the original list elements.

    Each derivative element is mapped onto a subset of original elements.
    The subsets must be disjoint.
    """

    list: List[Tuple[_T, Set[int]]] = field(default_factory=list)

    @property
    def raw_list(self) -> List[_T]:
        return [msg for msg, _ in self.list]

    def to_original_indices(self, indices: Iterable[int]) -> Set[int]:
        return {
            orig_index
            for index in indices
            for orig_index in self.list[index][1]
        }

    def append(self, elem: _T, idx: int) -> Self:
        self.list.append((elem, {idx}))
        return self


async def aiter_to_list(iterator: AsyncIterator[_T]) -> List[_T]:
    return [item async for item in iterator]
