import logging
import os
from typing import Any, Optional
from uuid import uuid4

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class JerichoInteraction(BaseInteraction):
    """Interaction for Jericho text adventure games.

    Each trajectory gets its own Jericho environment instance.
    The LLM generates one action per turn, and the environment
    returns the next observation and step reward.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict: dict[str, dict] = {}

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
        }
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        instance = self._instance_dict[instance_id]
        env = instance["env"]

        # Extract action from the last assistant message
        # Supports "ACTION: xxx" format (evotest style) and plain text
        action = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                lines = content.strip().split("\n")
                # First try to find "ACTION: xxx" line
                for line in lines:
                    if line.upper().startswith("ACTION:"):
                        action = line.split(":", 1)[1].strip()
                        break
                # Fallback: use last non-empty line
                if not action:
                    for line in reversed(lines):
                        line = line.strip()
                        if line:
                            action = line
                            break
                break

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

        response_text = obs.strip() if obs and obs.strip() else "Nothing happens."

        return done, response_text, reward, {
            "step_reward": reward,
            "total_reward": instance["total_reward"],
            "score": info.get("score", 0),
            "steps": instance["steps"],
            "action": action,
        }

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        if instance_id in self._instance_dict:
            return self._instance_dict[instance_id]["total_reward"]
        return 0.0

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            try:
                self._instance_dict[instance_id]["env"].close()
            except Exception:
                pass
            del self._instance_dict[instance_id]
