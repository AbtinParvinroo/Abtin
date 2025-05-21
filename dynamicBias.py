import numpy as np

def dynamic_bias_layer(input_feature_map, bias_weights):
    H, W, C_out = input_feature_map.shape
    assert bias_weights.shape[0] == C_out
    bias_reshaped = bias_weights.reshape(1, 1, C_out)
    output = input_feature_map + bias_reshaped
    return output