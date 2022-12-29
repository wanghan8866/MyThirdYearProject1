from typing import List, Dict, Set, Tuple


class AppMeta:
    APP_TITLE = "My Third Year Project"
    SIZE = "750x450"
    GAMES: Dict[str, List[str]] = {
        "Breakout": [
            "randomAgent",
            "randomAgent",
            "breakoutA2C",
            "breakoutA2C",
            "human"
        ],
        "BattleZone": [
            "randomAgent",
            "randomAgent",
            "randomAgent",
            "human"
        ],
        "BankHeist": [
            "randomAgent",
            "randomAgent",
            "human"
        ],
        "Tennis": [
            "randomAgent",
            "human"
        ],
        "Skiing": [
            "randomAgent",
            "human"
        ],
        "FlappyBird": [
            "randomAgent",
            "randomAgent",
            "human"
        ],

    }
    KEY_MAP: Dict[str, Tuple[str, int]] = {
        "space": ("FIRE", 1),
        "w": ("UP", 2),
        "d": ("RIGHT", 3),
        "a": ("LEFT", 4),
        "s": ("DOWN", 5),
        "e": ("UPRIGHT", 6),
        "q": ("UPLEFT", 7),
        "c": ("DOWNRIGHT", 8),
        "z": ("DOWNLEFT", 9),
        "8": ("UPFIRE", 10),
        "6": ("RIGHTFIRE", 11),
        "4": ("LEFTFIRE", 12),
        "2": ("DOWNFIRE", 13),
        "9": ("UPRIGHTFIRE", 14),
        "7": ("UPLEFTFIRE", 15),
        "3": ("DOWNRIGHTFIRE", 16),
        "1": ("DOWNLEFTFIRE", 17),
    }
    GAME_KEY_MAP: Dict = {
        "Breakout": {
            "FIRE": 1,
            "RIGHT": 2,
            "LEFT": 3
        },
        "Skiing": {
            "RIGHT": 1,
            "LEFT": 2
        },
        "FlappyBird": {
            "FIRE": 1,
        }

    }
