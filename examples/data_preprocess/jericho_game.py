"""Preprocess Jericho games to parquet format for verl multi-turn training."""

import argparse
import glob
import os

import pandas as pd


SYSTEM_PROMPT = (
    "You are an expert player aiming to complete a text-based adventure game. "
    "Points are given for making progress in the game. "
    "Select promising actions based on the game state and memory of past interactions."
)

USER_PROMPT_SUFFIX = """
Type your next action as if you were playing the game directly. It should be a short command that can be understood by the game parser. Common actions include: look, inventory, directions (north, northeast, up, etc.), examine X, say X, drop X, get X, open X, enter X, ask X about Y, look in X, give X to Y, and other context-specific commands. To avoid parsing errors, any such X or Y MUST be a *SINGLE WORD* that identifies an object in a way the game can understand. When stuck, explore all rooms and objects mentioned in room descriptions systematically and comprehensively. *DO NOT REPEAT* the same failed action multiple times, as it will not lead to new results. Do not use the "help" command.

Your response MUST strictly follow this format and include nothing else:
REASONING: [A short, concise explanation of your choice, 1-2 sentences]
ACTION: [short word or phrase for text command to execute]

For example:
REASONING: I should examine the book to learn more about it.
ACTION: examine book"""


def get_initial_observation(game_path: str) -> str:
    """Get the initial observation from a Jericho game."""
    import jericho

    env = jericho.FrotzEnv(game_path)
    obs, info = env.reset()
    env.close()
    return obs.strip()


def main():
    parser = argparse.ArgumentParser(description="Preprocess Jericho games to parquet")
    parser.add_argument(
        "--game_dir",
        default="jericho-games",
        help="Directory containing .z5 game files",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/jericho",
        help="Output directory for parquet files",
    )
    args = parser.parse_args()

    game_dir = os.path.abspath(args.game_dir)
    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    game_files = sorted(glob.glob(os.path.join(game_dir, "*.z5")))
    if not game_files:
        raise FileNotFoundError(f"No .z5 files found in {game_dir}")

    print(f"Found {len(game_files)} games: {[os.path.basename(f) for f in game_files]}")

    records = []
    for idx, game_path in enumerate(game_files):
        game_name = os.path.splitext(os.path.basename(game_path))[0]
        print(f"Processing {game_name}...")

        initial_obs = get_initial_observation(game_path)

        user_content = f"Your current state is: {initial_obs}\n{USER_PROMPT_SUFFIX}"

        record = {
            "data_source": "jericho",
            "agent_name": "tool_agent",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "ability": "game",
            "reward_model": {"style": "rule", "ground_truth": game_path},
            "extra_info": {
                "split": "train",
                "index": idx,
                "game_name": game_name,
                "game_file": game_path,
                "interaction_kwargs": {
                    "name": "jericho",
                    "game_file": game_path,
                },
            },
        }
        records.append(record)

    df = pd.DataFrame(records)

    train_path = os.path.join(save_dir, "train.parquet")
    test_path = os.path.join(save_dir, "test.parquet")
    df.to_parquet(train_path)
    df.to_parquet(test_path)

    print(f"Saved {len(records)} records to:")
    print(f"  Train: {train_path}")
    print(f"  Test:  {test_path}")


if __name__ == "__main__":
    main()
