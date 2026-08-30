"""Prime Intellect integration owned by this experiment.

The chat format is intentionally framework-neutral.  PrimeRL-specific source
registration is kept in :mod:`posttrain.prime.renderer` and the pinned patch
under ``integrations/prime_intellect``.
"""

from posttrain.prime.chat_format import (
    FORMAT_ID,
    EOS_TOKEN_ID,
    ChatFormatError,
    RenderedChat,
    StarCoder2CodingChatFormat,
    StarCoder2ChatFormat,
)

__all__ = [
    "FORMAT_ID",
    "EOS_TOKEN_ID",
    "ChatFormatError",
    "RenderedChat",
    "StarCoder2CodingChatFormat",
    "StarCoder2ChatFormat",
]
