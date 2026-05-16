import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        weights_sum=np.sum(x*w);
        def sigmoid(a):
            return 1/(1+np.exp(-a))
        def relu(a):
            return max(0.0,a)
        if activation=="sigmoid":
            return round(sigmoid(weights_sum+b),5)
        return round(relu(weights_sum+b),5)
