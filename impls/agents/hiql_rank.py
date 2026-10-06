"""Default configuration for the same-trajectory ranking variant of HIQL."""

from agents.hiql import get_config as get_hiql_config


def get_config():
    config = get_hiql_config()
    config.value_loss_type = 'rank'
    # Preserve the original one-sided hinge objective by default.
    config.rank_upper_weight = 0.0
    return config
