from models.coast import ThreeBranchesPretrain


def get_model(config, attributes=None, classes=None, offset=None, dset=None):
    if config.model_name in ["ThreeBranchesPretrain", "COAST"]:
        return ThreeBranchesPretrain(config, dset=dset)

    raise NotImplementedError(
        f"Unrecognized model name {config.model_name!r}. "
        "This GitHub release keeps only the COAST model code."
    )
