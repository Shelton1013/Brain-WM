"""Position-preserving handling of per-epoch annotation codes."""


def map_epoch_labels(lines, mapping):
    """Map nonempty annotation rows, retaining None for an unknown epoch.

    Filtering unknown labels before indexing the recording would shift all
    subsequent annotations. Consumers must skip the corresponding signal epoch.
    """
    return [mapping.get(line.strip()) for line in lines if line.strip()]
