"""Custom reward function for Jericho text adventure games."""

from typing import Any, Optional


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    **kwargs,
) -> float:
    if extra_info is None:
        return 0.0

    # Debug: print everything relevant
    turn_scores = extra_info.get("turn_scores", None)
    tool_rewards = extra_info.get("tool_rewards", None)
    rollout_reward_scores = extra_info.get("rollout_reward_scores", None)
    num_turns = extra_info.get("num_turns", None)

    print(f"[JERICHO_REWARD] turn_scores={turn_scores} (type={type(turn_scores).__name__})")
    print(f"[JERICHO_REWARD] tool_rewards={tool_rewards} (type={type(tool_rewards).__name__})")
    print(f"[JERICHO_REWARD] rollout_reward_scores={rollout_reward_scores}")
    print(f"[JERICHO_REWARD] num_turns={num_turns}")

    # Try turn_scores first
    if turn_scores is not None:
        try:
            scores = list(turn_scores)
            if scores:
                total = float(sum(scores))
                print(f"[JERICHO_REWARD] Using turn_scores sum={total}")
                return total
        except Exception as e:
            print(f"[JERICHO_REWARD] Error processing turn_scores: {e}")

    # Try tool_rewards
    if tool_rewards is not None:
        try:
            rewards = list(tool_rewards)
            if rewards:
                total = float(sum(rewards))
                print(f"[JERICHO_REWARD] Using tool_rewards sum={total}")
                return total
        except Exception as e:
            print(f"[JERICHO_REWARD] Error processing tool_rewards: {e}")

    # Fallback: replay from solution_str
    import os
    game_file = ground_truth
    if not game_file or not os.path.exists(game_file):
        print(f"[JERICHO_REWARD] No game file, returning 0")
        return 0.0

    # Extract actions - handle <think> tags from Qwen3
    text = solution_str
    # Remove <think>...</think> blocks
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    actions = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("ACTION:"):
            action = line.split(":", 1)[1].strip()
            if action:
                actions.append(action)

    print(f"[JERICHO_REWARD] Replay: {len(actions)} actions from solution_str")
    if not actions:
        return 0.0

    import jericho
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
        print(f"[JERICHO_REWARD] Replay reward={total_reward}")
        return total_reward
    except Exception as e:
        print(f"[JERICHO_REWARD] Replay error: {e}")
        return 0.0
