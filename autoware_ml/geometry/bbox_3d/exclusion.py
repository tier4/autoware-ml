# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Box exclusion rules by class and attribute, shared by the box filters, samplers and recipes."""

from collections.abc import Sequence


def parse_exclusion_rules(rules: Sequence[Sequence[str]]) -> frozenset[tuple[str, str]]:
    """
    Read box exclusion rules as class and attribute pairs.

    Args:
      rules: Exclusion rules, each a [class_name, attribute] pair.

    Returns:
      frozenset[tuple[str, str]]: The rules as (class_name, attribute) pairs.

    Raises:
      ValueError: If a rule is not a pair.
    """
    for index, rule in enumerate(rules):
        if isinstance(rule, str) or len(rule) != 2:
            raise ValueError(
                f"Exclusion rule {index} must be a [class_name, attribute] pair, got {rule!r}."
            )
    return frozenset((str(rule[0]), str(rule[1])) for rule in rules)


def excluded_boxes(
    label_names: Sequence[str],
    attributes: Sequence[Sequence[str]],
    rules: frozenset[tuple[str, str]],
) -> list[bool]:
    """
    Whether an exclusion rule names the class and an attribute of every box.

    Args:
      label_names: Fine label name of every box.
      attributes: Attributes of every box.
      rules: Exclusion rules from :func:`parse_exclusion_rules`.

    Returns:
      list[bool]: True for every box a rule excludes.
    """
    return [
        any((label_name, attribute) in rules for attribute in box_attributes)
        for label_name, box_attributes in zip(label_names, attributes, strict=True)
    ]
