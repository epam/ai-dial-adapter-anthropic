import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

_T = TypeVar("_T")


async def make_async(func: Callable[[], _T]) -> _T:
    with ThreadPoolExecutor(max_workers=1) as executor:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, func)
