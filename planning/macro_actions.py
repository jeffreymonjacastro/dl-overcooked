"""Macro-action names used by planners and future Macro-PPO."""

from __future__ import annotations

from enum import IntEnum


class MacroAction(IntEnum):
    GET_INGREDIENT = 0
    PLACE_INGREDIENT_IN_POT = 1
    GET_DISH = 2
    PICK_READY_SOUP = 3
    DELIVER_SOUP = 4
    PASS_INGREDIENT = 5
    PASS_DISH = 6
    RECEIVE_OBJECT = 7
    CLEAR_COUNTER = 8
    YIELD = 9
    UNBLOCK = 10
    WAIT_NEAR_TARGET = 11
    SOLO_PIPELINE = 12


MACRO_ACTIONS = tuple(MacroAction)

