"""Custom reward function for Jericho text adventure games.

Replays the LLM's actions in the Jericho environment to compute the game score.
Used via reward.custom_reward_function.path in the training config.
"""

import os
import re
from typing import Any, Optional


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    **kwargs,
) -> float:
    """Compute reward by replaying LLM actions in Jericho.

    Args:
        data_source: Should be "jericho".
        solution_str: The decoded full response text containing all assistant turns.
        ground_truth: The game file path (set in parquet data).
        extra_info: Additional info from the dataset.

    Returns:
        The total game score as a float.
    """
    import jericho

    game_file = ground_truth
    if not game_file or not os.path.exists(game_file):
        return 0.0

    # Parse actions from the multi-turn conversation text.
    # In multi-turn mode, solution_str contains the full decoded response
    # including both assistant and user turns. We extract lines that look
    # like short action commands (typically 1-5 words).
    actions = _extract_actions(solution_str)

    if not actions:
        return 0.0

    # Replay actions in Jericho
    try:
        env = jericho.FrotzEnv(game_file)
        env.reset()
        total_reward = 0.0
        for action in actions:
            try:
                obs, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break
            except Exception:
                continue
        env.close()
        return total_reward
    except Exception:
        return 0.0


def _extract_actions(solution_str: str) -> list[str]:
    """Extract action commands from the multi-turn response text.

    Looks for "ACTION: xxx" lines (evotest format). Falls back to
    short lines as a heuristic if no ACTION: lines are found.
    """
    actions = []
    lines = solution_str.strip().split("\n")

    # First pass: look for "ACTION: xxx" lines
    for line in lines:
        line = line.strip()
        if line.upper().startswith("ACTION:"):
            action = line.split(":", 1)[1].strip()
            if action:
                actions.append(action)

    if actions:
        return actions

    # Fallback: short lines that look like commands
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) <= 60 and line.count(".") <= 1:
            words = line.split()
            if 1 <= len(words) <= 8:
                actions.append(line)
    return actions
