import random
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Initialize emotion library
try:
    from reachy_mini.motion.recorded_move import RecordedMoves
    from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove

    # Note: huggingface_hub automatically reads HF_TOKEN from environment variables
    RECORDED_MOVES = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
    EMOTION_AVAILABLE = True
except Exception as e:
    logger.warning(f"Emotion library not available: {e}")
    RECORDED_MOVES = None
    EMOTION_AVAILABLE = False


def get_available_emotions_and_descriptions() -> str:
    """Get formatted list of available emotions with descriptions."""
    if not EMOTION_AVAILABLE:
        return "Emotions not available"

    try:
        emotion_names = RECORDED_MOVES.list_moves()
        if not emotion_names:
            return "No emotions currently available"

        output = "Available emotions:\n"
        for name in emotion_names:
            description = RECORDED_MOVES.get(name).description
            output += f" - {name}: {description}\n"
        return output
    except Exception as e:
        return f"Error getting emotions: {e}"


_SEMANTIC_EMOTION_CANDIDATES = {
    "happy": ("happy", "smile", "joy", "glad", "love", "heart", "cute", "excited", "yeah", "yes"),
    "smile": ("smile", "happy", "joy", "cute", "love", "heart"),
    "curious": ("curious", "question", "wonder", "think", "oops"),
    "oops": ("oops", "surprise", "shy"),
}


def _choose_emotion(emotion_name: Any, emotion_names: list[str]) -> str:
    """Resolve semantic aliases to a concrete recorded emotion name."""
    if not emotion_names:
        return ""

    requested = str(emotion_name or "").strip()
    if requested in emotion_names:
        return requested

    lowered = {name.lower(): name for name in emotion_names}
    requested_lower = requested.lower()
    candidates = _SEMANTIC_EMOTION_CANDIDATES.get(requested_lower, ())
    for keyword in candidates:
        for lower_name, original_name in lowered.items():
            if keyword in lower_name:
                return original_name

    # Random emotion often feels wrong for explicit smile requests. Prefer a positive move by default.
    for keyword in _SEMANTIC_EMOTION_CANDIDATES["happy"]:
        for lower_name, original_name in lowered.items():
            if keyword in lower_name:
                return original_name

    if requested:
        raise ValueError(f"Unknown emotion '{requested}'. Available: {emotion_names}")
    return random.choice(emotion_names)


class PlayEmotion(Tool):
    """Play a pre-recorded emotion."""

    name = "play_emotion"
    description = "Play a pre-recorded emotion"
    parameters_schema = {
        "type": "object",
        "properties": {
            "emotion": {
                "type": "string",
                "enum": list(RECORDED_MOVES.list_moves()) if EMOTION_AVAILABLE else [],
                "description": f"""Name of the emotion to play; omit for random.
                                    Here is a list of the available emotions, you MUST only choose from these: \n
                                    {get_available_emotions_and_descriptions()}
                                    """,
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Play a pre-recorded emotion."""
        if not EMOTION_AVAILABLE:
            return {"error": "Emotion system not available"}

        emotion_name = kwargs.get("emotion")

        logger.info("Tool call: play_emotion emotion=%s", emotion_name)

        # Check if emotion exists
        try:
            emotion_names = RECORDED_MOVES.list_moves()
            if not emotion_names:
                return {"error": "No emotions currently available"}

            try:
                emotion_name = _choose_emotion(emotion_name, emotion_names)
            except ValueError as exc:
                return {"error": str(exc)}

            # Add emotion to queue
            movement_manager = deps.movement_manager
            emotion_move = EmotionQueueMove(emotion_name, RECORDED_MOVES)
            movement_manager.queue_move(emotion_move)

            return {"status": "queued", "emotion": emotion_name}

        except Exception as e:
            logger.exception("Failed to play emotion")
            return {"error": f"Failed to play emotion: {e!s}"}
