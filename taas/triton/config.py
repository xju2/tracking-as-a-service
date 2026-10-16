"""Triton model-configuration accessors."""


def get_parameter(model_config, name, default):
    return model_config.get("parameters", {}).get(
        name, {"string_value": str(default)}
    )["string_value"]
