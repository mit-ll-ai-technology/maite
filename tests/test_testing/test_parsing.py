# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# flake8: noqa

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from maite._internals.testing.pyright import md_to_code, rst_to_code

src1 = """.. tab-set::

   .. tab-item:: dataclass-based config

      .. code-block:: python
         :caption: aa
      
         from dataclasses import dataclass
      
         @dataclass
         class Builds_DNN:
             input_size: int
             output_size: int
             layer_widths: tuple[int, ...] = (5, 10, 5)
             device: str = "cpu"
             _target_: str = "vision.model.DNN"

   .. tab-item:: yaml-based config

      .. code-block:: yaml
         :caption: blah
      
         _target_: vision.model.DNN
         input_size: ???
         output_size: ???
         layer_widths:
         - 5
         - 10
         - 5
"""
expected1 = """
from dataclasses import dataclass

@dataclass
class Builds_DNN:
    input_size: int
    output_size: int
    layer_widths: tuple[int, ...] = (5, 10, 5)
    device: str = "cpu"
    _target_: str = "vision.model.DNN"

"""

src2 = """
.. code-block:: pycon

   >>> from pathlib import Path 
   >>> def print_file(x: int):
   ...     with x.open("r") as f: 
   ...         print(f.read())
"""

src3 = """
    .. code-block:: python
       :caption: blah
       :foo: a
       :bar: b

       from pathlib import Path
   
       def print_file(x: Path):
           with x.open("r") as f: 
               print(f.read())
"""

expected2 = """from pathlib import Path 

def print_file(x: int):
    with x.open("r") as f: 
        print(f.read())
"""
expected3 = """
from pathlib import Path

def print_file(x: Path):
    with x.open("r") as f: 
        print(f.read())
"""


@pytest.mark.parametrize(
    "src,expected",
    [
        pytest.param(src1, expected1, id="src1"),
        pytest.param(src2, expected2, id="src2"),
        pytest.param(src2 * 3, "\n".join([expected2] * 3), id="src2 repeat"),
        pytest.param(src3 * 3, "\n".join([expected3] * 3), id="src3 repeat"),
        pytest.param(src1 + src2, "\n".join([expected1, expected2]), id="src1+src2"),
        pytest.param("", "", id="empty input"),
    ],
)
def test_rst_parsing_manual(src: str, expected: str):
    assert strip_interspacing(rst_to_code(src)) == strip_interspacing(expected)


distractors = [
    "",
    "\n",
    "python",
    "pycon",
    ">>> x = 1",
    "word.",
    ".. code-block::",
    "  .. code-block::",
    """.. code-block:: shell

   blah blah""",
    """.. code-block:: shell
   :caption: hello

   blah blah""",
]


def strip_interspacing(x: str):
    return "\n".join(s for s in x.splitlines() if s)


import textwrap


@settings(max_examples=500)
@given(
    blocks=st.lists(
        st.sampled_from([(src1, expected1), (src2, expected2), (src3, expected3)]),
        min_size=1,
        max_size=5,
    ),
    data=...,
    indent_level=st.integers(0, 9),
)
def test_rst_parsing(
    blocks: list[tuple[str, str]], data: st.DataObject, indent_level: int
):
    """Combines known src blocks (with associated expected outputs), joined by
    different patterns of joining text"""
    src_blocks = [b[0] for b in blocks]
    expected_blocks = [b[1] for b in blocks]

    num_blocks = len(blocks)
    dummy_blocks = data.draw(
        st.lists(
            st.sampled_from(distractors),
            min_size=num_blocks + 1,
            max_size=num_blocks + 1,
        )
    )
    src = "\n".join(a + b for a, b in zip(dummy_blocks, src_blocks + [""]))
    src = textwrap.indent(src, " " * indent_level)
    expected = strip_interspacing("\n".join([s.strip() for s in expected_blocks]))
    out = strip_interspacing(rst_to_code(src))

    assert out == expected


md_src1 = """.. tab-set::

   .. tab-item:: dataclass-based config

      ```python
      
         from dataclasses import dataclass
      
         @dataclass
         class Builds_DNN:
             input_size: int
             output_size: int
             layer_widths: tuple[int, ...] = (5, 10, 5)
             device: str = "cpu"
             _target_: str = "vision.model.DNN"
      ```

   ```yaml
         _target_: vision.model.DNN
         input_size: ???
         output_size: ???
         layer_widths:
         - 5
         - 10
         - 5
   ```
"""
md_expected1 = """
from dataclasses import dataclass

@dataclass
class Builds_DNN:
    input_size: int
    output_size: int
    layer_widths: tuple[int, ...] = (5, 10, 5)
    device: str = "cpu"
    _target_: str = "vision.model.DNN"

"""

md_src2 = """
```pycon

   >>> from pathlib import Path 
   >>> def print_file(x: int):
   ...     with x.open("r") as f: 
   ...         print(f.read())
```
"""

md_src3 = """
    ````
    an example
    ```python
    x = 2
    ```
    ````
    ```python

       from pathlib import Path
   
       def print_file(x: Path):
           with x.open("r") as f: 
               print(f.read())
    ```
"""

md_expected2 = """from pathlib import Path 

def print_file(x: int):
    with x.open("r") as f: 
        print(f.read())
"""
md_expected3 = """
from pathlib import Path

def print_file(x: Path):
    with x.open("r") as f: 
        print(f.read())
"""


@pytest.mark.parametrize(
    "src,expected",
    [
        pytest.param(md_src1, md_expected1, id="src1"),
        pytest.param(md_src2, md_expected2, id="src2"),
        pytest.param(md_src2 * 3, "\n".join([md_expected2] * 3), id="src2 repeat"),
        pytest.param(md_src3 * 3, "\n".join([md_expected3] * 3), id="src3 repeat"),
        pytest.param(
            md_src1 + md_src2, "\n".join([md_expected1, md_expected2]), id="src1+src2"
        ),
        pytest.param("", "", id="empty input"),
    ],
)
def test_md_parsing_manual(src: str, expected: str):
    assert strip_interspacing(md_to_code(src)) == strip_interspacing(expected)
