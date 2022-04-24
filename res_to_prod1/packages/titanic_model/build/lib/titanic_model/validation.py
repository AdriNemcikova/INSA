from titanic_model.config import config


def validate_inputs(input_data):
    validated_data = input_data.copy()

    # vsetky hodnotu musia byt kladne
    if (input_data[config.NUMERICALS_VARS] < 0).any().any():
        vars_with_neg_values = config.NUMERICALS_VARS[
            (input_data[config.NUMERICALS_VARS] < 0).any()
        ]
        validated_data = validated_data[validated_data[vars_with_neg_values] > 0]

    # nepovolene atributy s NA
    if input_data[config.NA_NOT_ALLOWED].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=config.NA_NOT_ALLOWED
        )

    return validated_data
