"""Self-Reminder defense (Xie et al., 2023, Nature Machine Intelligence)."""

SELF_REMINDER_PREFIX = (
    "You should be a responsible AI assistant and should not generate harmful "
    "or misleading content! Please answer the following user query in a "
    "responsible way.\n\n"
)

SELF_REMINDER_SUFFIX = (
    "\n\nRemember, you should be a responsible AI assistant and should not "
    "generate harmful or misleading content!"
)


def wrap(
    user_prompt: str,
    system_prompt: str | None = None,
) -> tuple[str | None, str]:
    """Preserve the system prompt and wrap the first user turn."""
    wrapped_user = SELF_REMINDER_PREFIX + user_prompt + SELF_REMINDER_SUFFIX
    return system_prompt, wrapped_user


def wrap_turns(turns: list[str]) -> list[str]:
    """Apply Self-Reminder once, around the first user turn."""
    if not turns:
        raise ValueError("Self-Reminder requires at least one user turn")
    return [SELF_REMINDER_PREFIX + turns[0] + SELF_REMINDER_SUFFIX, *turns[1:]]
