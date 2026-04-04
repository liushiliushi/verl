import json
import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

ACTION_PROMPT_SUFFIX = """
Your response MUST strictly follow this format and include nothing else:
REASONING: [A short, concise explanation of your choice, 1-2 sentences]
ACTION: [short word or phrase for text command to execute]"""

# Directory to dump rollout traces
ROLLOUT_DUMP_DIR = os.path.expanduser("~/data/jericho_rollout_traces")


class JerichoInteraction(BaseInteraction):
    """Interaction for Jericho text adventure games."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict: dict[str, dict] = {}
        os.makedirs(ROLLOUT_DUMP_DIR, exist_ok=True)

    async def start_interaction(
        self, instance_id: Optional[str] = None, game_file: Optional[str] = None, **kwargs
    ) -> str:
        import jericho

        if instance_id is None:
            instance_id = str(uuid4())

        if game_file is None:
            raise ValueError("game_file is required in interaction_kwargs")

        if not os.path.exists(game_file):
            raise FileNotFoundError(f"Game file not found: {game_file}")

        env = jericho.FrotzEnv(game_file)
        obs, info = env.reset()

        self._instance_dict[instance_id] = {
            "env": env,
            "game_file": game_file,
            "total_reward": 0.0,
            "steps": 0,
            "trace": [],  # record each turn for debugging
        }
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        instance = self._instance_dict[instance_id]
        env = instance["env"]

        # Extract action from the last assistant message
        action = self._extract_action(messages)

        if not action:
            action = "look"

        # Execute action in Jericho environment
        try:
            obs, reward, done, info = env.step(action)
        except Exception as e:
            logger.warning(f"Jericho step failed for action '{action}': {e}")
            obs = "Nothing happens."
            reward = 0.0
            done = False
            info = {}

        reward = float(reward)
        instance["total_reward"] += reward
        instance["steps"] += 1

        obs_text = obs.strip() if obs and obs.strip() else "Nothing happens."

        # Get valid actions from Jericho
        valid_actions = []
        try:
            valid_actions = env.get_valid_actions()
        except Exception:
            pass

        # Build user response: observation + valid actions + format reminder
        if valid_actions:
            valid_str = ", ".join(valid_actions[:20])
            response_text = f"{obs_text}\n\nValid actions: {valid_str}{ACTION_PROMPT_SUFFIX}"
        else:
            response_text = f"{obs_text}{ACTION_PROMPT_SUFFIX}"

        # Get the last assistant message for trace
        last_assistant_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_assistant_msg = msg.get("content", "")
                break

        # Record trace
        instance["trace"].append({
            "step": instance["steps"],
            "assistant_response": last_assistant_msg,
            "extracted_action": action,
            "observation": obs_text,
            "valid_actions": valid_actions[:20] if valid_actions else [],
            "reward": reward,
            "total_reward": instance["total_reward"],
            "score": info.get("score", 0),
            "done": done,
        })

        print(f"[JERICHO] step={instance['steps']} action='{action}' reward={reward} total={instance['total_reward']} score={info.get('score', 0)}")

        return done, response_text, reward, {
            "step_reward": reward,
            "total_reward": instance["total_reward"],
            "score": info.get("score", 0),
            "steps": instance["steps"],
            "action": action,
        }

    def _extract_action(self, messages: list[dict[str, Any]]) -> str:
        """Extract action from the last assistant message, handling <think> tags."""
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")

            # Strip <think>...</think> blocks (Qwen3 thinking mode)
            content_clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

            lines = content_clean.strip().split("\n")

            # First try to find "ACTION: xxx" line
            for line in lines:
                line = line.strip()
                if line.upper().startswith("ACTION:"):
                    action = line.split(":", 1)[1].strip()
                    if action:
                        return action

            # Fallback: last non-empty short line (likely a command)
            for line in reversed(lines):
                line = line.strip()
                if line and len(line) <= 60:
                    return line

            break
        return ""

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        if instance_id in self._instance_dict:
            return self._instance_dict[instance_id]["total_reward"]
        return 0.0

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            instance = self._instance_dict[instance_id]
            # Dump trace to file
            try:
                trace_file = os.path.join(ROLLOUT_DUMP_DIR, f"{instance_id}.json")
                with open(trace_file, "w") as f:
                    json.dump({
                        "game_file": instance["game_file"],
                        "total_reward": instance["total_reward"],
                        "total_steps": instance["steps"],
                        "trace": instance["trace"],
                    }, f, indent=2, ensure_ascii=False)
                print(f"[JERICHO] Trace saved to {trace_file} (steps={instance['steps']}, reward={instance['total_reward']})")
            except Exception as e:
                print(f"[JERICHO] Failed to save trace: {e}")
            try:
                instance["env"].close()
            except Exception:
                pass
            del self._instance_dict[instance_id]
