# Copyright (c) 2025 BeThere AI
# All rights reserved.
#
from enum import Enum

OwnerMode = Enum('OwnerMode', ["normal", "extended", "negative"])


# This considers the match under the exact sense.
class ExactMatcher:
    @staticmethod
    def agree(owner, owner_mode, target, target_mode):
        label_match = owner == target
        if not label_match:
            return label_match

        # We should not use this example.
        if OwnerMode[owner_mode] != OwnerMode.normal and OwnerMode[target_mode] != OwnerMode.normal:
            return None

        # now we have match, but mode does not match.
        return OwnerMode[owner_mode] == OwnerMode.normal and OwnerMode[target_mode] == OwnerMode.normal

    @staticmethod
    def match(owner, target, mode_in_str):
        return owner == target and OwnerMode[mode_in_str] == OwnerMode.normal

    @staticmethod
    def is_good_mode(mode_in_str):
        return OwnerMode[mode_in_str] == OwnerMode.normal