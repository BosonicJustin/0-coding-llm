"""Post-training data and framework adapters.

Post-training artifacts intentionally do not reuse the pre-training packed
format.  SFT needs role-aware loss masks and whole conversations must remain
intact, while RL additionally needs runtime and reward metadata.
"""
