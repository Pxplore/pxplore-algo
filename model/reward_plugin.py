import asyncio
import os
import re
import textwrap
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Optional

import json
import torch

from swift.llm import PtEngine, RequestConfig, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse
from swift.plugin import ORM, orms, rm_plugins
# register context manager(used in gym training)
from swift.plugin.context_manager import ContextManager, context_managers
from swift.plugin.env import Env, envs
from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
from swift.plugin.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger

logger = get_logger()

def compute_kl(policy_logits, ref_logits, mask=None):
    """
    Compute average KL divergence between policy and reference model.
    Args:
        policy_logits: [seq_len, vocab_size] or [batch, seq_len, vocab_size]
        ref_logits: same shape as policy_logits
        mask: optional [seq_len] or [batch, seq_len]
    """
    logp = F.log_softmax(policy_logits, dim=-1)
    logq = F.log_softmax(ref_logits, dim=-1)
    kl = torch.exp(logp) * (logp - logq)   # per token KL
    kl = kl.sum(-1)                        # sum over vocab

    if mask is not None:
        kl = kl * mask
        return kl.sum() / mask.sum()
    return kl.mean()

class PxploreRewardORM(ORM):
    """
    Pxplore Reward Model that calculates learning state alignment scores.
    Evaluates how well the next lesson content aligns with student's learning objectives 
    across four dimensions: long_term_objective, short_term_objective, implicit_motivation, explicit_motivation.
    """

    def __init__(self, kl_coef=0.1):
        # Try to import required dependencies
        try:
            from pathlib import Path
            import sys
            import os
            
            # Add the project root to Python path for imports
            # First try using the current working directory (when run from pxplore-algo)
            cwd = Path.cwd()
            if (cwd / "service" / "llm" / "openai.py").exists() and (cwd / "data" / "snippet.py").exists():
                project_root = cwd
            else:
                # Fallback to calculating from file path
                project_root = Path(__file__).parent.parent.parent.parent.parent.parent
            
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            from service.llm.openai import OPENAI_SERVICE
            from data.snippet import get_snippet
            
            self.openai_service = OPENAI_SERVICE
            self.get_snippet = get_snippet
            
            # Load the reward prompt
            prompt_path = Path(__file__).parent.parent / "qwen3" / "reward_prompt.txt"
            self.prompt = open(prompt_path, "r", encoding="utf-8").read()

            self.kl_coef = kl_coef
            
        except ImportError as e:
            logger.warning(f"Failed to import Pxplore dependencies: {e}. PxploreRewardORM may not work properly.")
            self.openai_service = None
            self.get_snippet = None
            self.prompt = ""
        except FileNotFoundError as e:
            logger.warning(f"Failed to load reward prompt file: {e}. PxploreRewardORM may not work properly.")
            self.openai_service = None
            self.get_snippet = None
            self.prompt = ""

    def call_reward_model(self, initial_state: dict, next_lesson_content: str) -> dict:
        """Call the reward model to get updated state."""
        if not self.openai_service:
            logger.warning("OPENAI_SERVICE not available, returning empty state")
            return {}
            
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": json.dumps({"初始学习状态": initial_state, "下一课内容": next_lesson_content})},
        ]

        job_id = self.openai_service.trigger(
            parent_service="Pxplore-Reward",
            parent_job_id=None,
            use_cache=True,
            model="gpt-4o",
            messages=messages
        )

        response = self.openai_service.get_response_sync(job_id)
        response = self.openai_service.parse_json_response(response)
        return response

    def extract_confidence(self, description: str) -> float:
        """Extract confidence value from description string."""
        import re
        try:
            # Look for confidence:X.XX pattern at the end of the description
            match = re.search(r'confidence:\s*([0-9]*\.?[0-9]+)', description)
            if match:
                return float(match.group(1))
            else:
                logger.warning(f"No confidence value found in description: {description}...")
                return 1
        except (ValueError, AttributeError) as e:
            logger.warning(f"Error extracting confidence from description: {e}")
            return 1

    def calculate_reward(self, next_state: dict) -> float:
        """Calculate reward based on aligned items * confidence values."""
        total_reward = 0.0
        total_items = 0
        
        dimension_stats = {
            'long_term_objective': {'total': 0, 'aligned_confidence_sum': 0.0, 'aligned_count': 0},
            'short_term_objective': {'total': 0, 'aligned_confidence_sum': 0.0, 'aligned_count': 0},
            'implicit_motivation': {'total': 0, 'aligned_confidence_sum': 0.0, 'aligned_count': 0},
            'explicit_motivation': {'total': 0, 'aligned_confidence_sum': 0.0, 'aligned_count': 0}
        }
        
        for dimension in dimension_stats.keys():
            if dimension in next_state and isinstance(next_state[dimension], list):
                for objective in next_state[dimension]:
                    if isinstance(objective, dict) and 'is_aligned' in objective and 'description' in objective:
                        dimension_stats[dimension]['total'] += 1
                        total_items += 1
                        
                        if objective['is_aligned']:
                            confidence = self.extract_confidence(objective['description'])
                            dimension_stats[dimension]['aligned_confidence_sum'] += confidence
                            dimension_stats[dimension]['aligned_count'] += 1
                            total_reward += confidence

        # Log detailed stats for debugging
        logger.info("Reward calculation details:")
        for dimension, stats in dimension_stats.items():
            if stats['total'] > 0:
                avg_confidence = stats['aligned_confidence_sum'] / stats['aligned_count'] if stats['aligned_count'] > 0 else 0.0
                logger.info(f"  {dimension}: {stats['aligned_count']}/{stats['total']} aligned, "
                           f"confidence sum: {stats['aligned_confidence_sum']:.3f}, "
                           f"avg confidence: {avg_confidence:.3f}")

        logger.info(f"Total reward: {total_reward / total_items:.3f} (from {total_items} total items)")
        return total_reward / total_items

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        Reward function that evaluates learning state alignment,
        with KL penalty to keep close to reference model.
        """
        rewards = []
        num_samples = len(completions)

        if not self.openai_service:
            return [0.0] * num_samples

        # 从 kwargs 拿 logits
        policy_logits = kwargs.get("policy_logits", None)
        ref_logits = kwargs.get("ref_logits", None)

        for i in range(num_samples):
            try:
                completion = completions[i]

                # 提取 initial_state
                try:
                    messages = kwargs.get('messages', [])
                    if i < len(messages) and len(messages[i]) > 1:
                        message_content = messages[i][1].get('content', '{}')
                        if isinstance(message_content, str):
                            message_data = json.loads(message_content)
                        else:
                            message_data = message_content
                        initial_state = message_data.get("recommendation_strategy", {})
                    else:
                        initial_state = {}
                except Exception:
                    initial_state = {}

                # 解析 completion
                try:
                    if "</think>" in completion:
                        completion_content = completion.split("</think>")[1].strip()
                    else:
                        completion_content = completion.strip()
                    output_json = json.loads(completion_content)
                except Exception:
                    logger.warning(f"Error parsing completion: {completion[:100]}...")
                    rewards.append(-1.0)
                    continue

                # 提取 lesson_id
                lesson_id = None
                selected_content = output_json.get("selected_content", {})
                if isinstance(selected_content, dict):
                    lesson_id = selected_content.get("id")
                    if not lesson_id and "metadata" in selected_content:
                        metadata = selected_content.get("metadata", {})
                        if isinstance(metadata, dict):
                            lesson_id = metadata.get("id")
                if not lesson_id:
                    selected_candidate = output_json.get("selected_candidate", {})
                    if isinstance(selected_candidate, dict):
                        lesson_id = selected_candidate.get("id")

                if not lesson_id:
                    logger.warning(f"No lesson id found in completion: {completion[:100]}...")
                    rewards.append(0.0)
                    continue

                # 获取 lesson 内容
                if self.get_snippet:
                    lesson_data = self.get_snippet(lesson_id)
                    if lesson_data and 'children' in lesson_data:
                        current_lesson_content = "\n".join(
                            [lesson['children'][1]["script"] for lesson in lesson_data["children"]]
                        )
                    else:
                        logger.warning(f"No lesson data found in completion: {completion[:100]}...")
                        rewards.append(0.0)
                        continue
                else:
                    logger.warning(f"No get_snippet function found in completion: {completion[:100]}...")
                    rewards.append(0.0)
                    continue

                if not current_lesson_content:
                    logger.warning(f"No current lesson content found in completion: {completion[:100]}...")
                    rewards.append(0.0)
                    continue

                # 调 reward model
                next_state = self.call_reward_model(initial_state, current_lesson_content)
                reward = self.calculate_reward(next_state)

                # === KL penalty ===
                if policy_logits is not None and ref_logits is not None:
                    try:
                        kl_value = compute_kl(policy_logits[i], ref_logits[i])
                        reward = reward - self.kl_coef * kl_value.item()
                    except Exception as e:
                        logger.warning(f"Error computing KL: {e}")

                rewards.append(reward)
                logger.info(f"Reward: {reward}")

            except Exception as e:
                logger.warning(f"Error: {e}")
                rewards.append(0.0)

        return rewards

orms['external_pxplore_reward'] = PxploreRewardORM
