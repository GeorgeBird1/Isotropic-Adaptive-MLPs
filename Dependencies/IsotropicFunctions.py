import torch
from torch import nn
import math

class IsotropicTanhIntrinsicLength(nn.Module):
    def __init__(self, trainable_intrinsic_length: bool = True, positive_intrinsic_length: bool = True,
                 init_intrinsic_length: float = 1e-6, epsilon: float = 1e-3,
                 device = None, dtype = None,
                 ):
        """
        This implements the isotropic tanh formula with the intrinsic length included as a parameterisation
        :param trainable_intrinsic_length: Whether the intrinsic length "o" is trainable by the optimiser.
        :param positive_intrinsic_length: Whether the intrinsic length "o" is always positive.
        :param init_intrinsic_length: The intrinsic value for the intrinsic length
        :param epsilon: Isotropic tanh is computed piecewise to ensure proper handling of coordinate singularity, this is the ball defined by R<epsilon condition for the piecewise approximation.
        standard device and dtype approach is used
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        # Add arguments to class
        self.epsilon = epsilon
        self.positive = positive_intrinsic_length
        self.trainable = trainable_intrinsic_length

        if self.trainable:
            if self.positive:
                # Set intrinsic length as a positive trainable parameter
                self.intrinsic_length = nn.Parameter(
                    torch.tensor(math.log(init_intrinsic_length), **factory_kwargs))
            else:
                # Set intrinsic length as a real trainable parameter
                self.intrinsic_length = nn.Parameter(
                    torch.tensor(init_intrinsic_length, **factory_kwargs))
        else:
            # Set intrinsic length as a positive non-trainable parameter
            self.register_buffer("o_const", torch.tensor(init_intrinsic_length, **factory_kwargs))

    @property
    def o(self):
        if self.trainable:
            if self.positive:
                return torch.exp(self.intrinsic_length)
            else:
                return self.intrinsic_length
        else:
            return self.o_const

    def _o_device(self):
        if self.trainable:
            return self.intrinsic_length.device
        else:
            return self.o_const.device

    @o.setter
    def o(self, value: float):
        """
        Sets the value of the intrinsic length
        - Trainable: updates in-place, no grad.
        - Non-trainable: updates the buffer.
        """

        # Convert to tensor if float or int
        if isinstance(value, (float, int)):
            value_t = torch.tensor(float(value), dtype=torch.get_default_dtype(), device=self._o_device())
        elif isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Must be a tensor-scalar/float/int!")
            value_t = value.to(dtype=torch.get_default_dtype(), device=self._o_device())
        else:
            raise TypeError("`o` must be a float, int, or scalar torch.Tensor.")

        if self.trainable:
            with torch.no_grad():
                if self.positive:
                    self.intrinsic_length.copy_(value_t.log())
                else:
                    self.intrinsic_length.copy_(value_t)
        else:
            self.o_const = value_t

    def forward(self, x: torch.Tensor, dims: tuple[int, ...] = (-1,)) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of isotropic tanh, given that tensor x is of shape [BATCH, 1, ..., n-1], but returns the non-linear term and the linear term separately
        :param x: The input tensor
        :param dims: Which dimensions to apply isotropic tanh over
        :return:
        """
        # Calculate the vector magnitude with intrinsic length
        r = torch.sum(torch.square(x), dim=dims, keepdim=True)
        magnitude = torch.sqrt(torch.abs(self.o + r))

        # A small ball approximates the identity map for more stable computation
        identity_mask = magnitude <= self.epsilon

        # The non-linear \left\|\vec{x}\right\|<epsilon approximation this time is more complicated, but it is the jacobian evaluated at zero for tanh(\left\|\vec{x};o\right\|\right)/\left\|\vec{x};o\right\|, as a local linear approximation. It results in \tanh{\sqrt{o}}/\sqrt{i}\mathrm{I}
        factor = torch.where(torch.abs(self.o) > 0.01, torch.tanh(torch.sqrt(torch.abs(self.o)))/torch.sqrt(torch.abs(self.o)), torch.ones_like(self.o))
        non_linear_term = torch.where(identity_mask, factor, torch.tanh(magnitude) / magnitude.clamp(min=self.epsilon))
        linear_term = x
        return non_linear_term, linear_term

    def combined(self, x: torch.Tensor, dims: tuple[int, ...] = (-1,)) -> torch.Tensor:
        """
        Performs the forward pass of isotropic tanh, given that tensor x is of shape [BATCH, 1, ..., n-1]
        :param x: The input tensor
        :param dims: Which dimensions to apply isotropic tanh over
        :return:
        """
        # Calculate the vector magnitude with intrinsic length
        r = torch.sum(torch.square(x), dim=dims, keepdim=True)
        magnitude = torch.sqrt(torch.abs(self.o + r))

        # A small ball approximates the identity map for more stable computation
        identity_mask = magnitude <= self.epsilon

        # Calculate the required unit-vector
        unit_vector = x / magnitude.clamp(min=self.epsilon)

        # Return final computation. Compared to the paper, this is computed as the sigma(\left\|\vec{x}\right\|)\hat{x} form rather than \tilde{\sigma}\left(\left\|\vec{x}\right\|\right)\vec{x} but are equivalent otherwise
        factor = torch.where(torch.abs(self.o) > 0.01, torch.tanh(torch.sqrt(torch.abs(self.o)))/torch.sqrt(torch.abs(self.o)), torch.ones_like(self.o))

        return torch.where(identity_mask, factor*x, torch.tanh(magnitude) * unit_vector)


class IsotropicTanhMLP(nn.Module):
    def __init__(self,
            layers: list[int],
            flatten: bool = True,
            unflatten_shape: tuple[int, ...] | None = None,

            intrinsic_length_approach: str = "None",
            linear_correction_approach: str = "None",

            positive_intrinsic_length: bool = True,
            init_intrinsic_length: float = 1e-6,
            tanh_epsilon: float = 1e-3,

            device = None, dtype = None,
            ):
        """
        Implements an MLP network with isotropic tanh activation function, and dynamic topologies implemented. It is assumed that b* and W* are zero initialised and thus undergo spontaneous symmetry breaking.
        :param layers: List specifying per layer widths, e.g. [784, 50, 50, 10]
        :param flatten: Whether to use flatten or unflatten the input.
        :param unflatten_shape: Specifies shape to unflatten the output
        :param intrinsic_length_approach: To be selected from "None", "Constant", "Trainable" --- indicating absence of the intrinsic length, constant intrinsic length, or trainable intrinsic length, respectively.
        :param linear_correction_approach: This is the psi vector from the paper, and the approach follows similarly from "None", "Constant", "Trainable", "Trainable+Decay", "Expectation".
        These are no correction, a non-trainable constant, a trainable constant, a trainable constant regularised to decay to zero, and the linear approximation to the bias instead of a psi vector.

        The following arguments are used by IsotropicTanhIntrinsicLength:
        positive_intrinsic_length, init_intrinsic_length, tanh_epsilon

        standard device and dtype approach is used
        """
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.unflatten_shape = unflatten_shape
        super().__init__()

        # Whether to also transform optimiser moments:
        self.transform_first_moment: bool = True
        self.approx_transform_second_moment: bool = True # diag(R diag(v) R^T) results in v'_{il} = R_{ij} v_j \delta_{jk} R_{lk} -> v'_i = R_{ij} R_{ij} v_j

        # Ensure the layers specification is correct (must be list or tuple, and contain ≥ 2 entries)
        if type(layers) is not list and type(layers) is not tuple:
            raise ValueError("\"layers\" must be a list or tuple")
        if len(layers) < 2: raise ValueError("\"layers\" must have at least 2 elements")
        if type(flatten) is not bool: raise ValueError("\"flatten\" must be a boolean")
        else: self.flatten = flatten

        # Implement specified intrinsic length approach
        self.intrinsic_length_approach = intrinsic_length_approach
        if type(self.intrinsic_length_approach) is not str: raise ValueError(
            "\"intrinsic_length_approach\" must be a string")
        match self.intrinsic_length_approach.upper():
            case "NONE":
                self.activation = nn.ModuleList([IsotropicTanhIntrinsicLength(False, False, 0.0, tanh_epsilon, device=device, dtype=dtype) for i in range(len(layers)-2)])
            case "CONSTANT":
                self.activation = nn.ModuleList([IsotropicTanhIntrinsicLength(False, positive_intrinsic_length, init_intrinsic_length, tanh_epsilon, device=device, dtype=dtype) for i in range(len(layers)-2)])
            case "TRAINABLE":
                self.activation = nn.ModuleList([IsotropicTanhIntrinsicLength(True, positive_intrinsic_length, init_intrinsic_length, tanh_epsilon, device=device, dtype=dtype) for i in range(len(layers)-2)])
            case _:
                raise ValueError(
                    f"Invalid intrinsic length approach, \"{self.intrinsic_length_approach.upper()}\" choose from \"None\", \"Constant\", or \"Trainable\"")

        # self.weight_parameters = [nn.Parameter(torch.empty((out_features, in_features), **self.factory_kwargs)) for in_features, out_features in zip(layers[:-1], layers[1:])]
        # self.bias_parameters = [nn.Parameter(torch.empty(out_features, **self.factory_kwargs)) for out_features in layers[1:]]


        self.weight_parameters = nn.ParameterList([
            nn.Parameter(torch.empty((out_features, in_features), **self.factory_kwargs))
            for in_features, out_features in zip(layers[:-1], layers[1:])
        ])

        self.bias_parameters = nn.ParameterList([
            nn.Parameter(torch.empty(out_features, **self.factory_kwargs))
            for out_features in layers[1:]
        ])

        # Implement specified linear correction approach
        self.linear_correction_approach = linear_correction_approach

        self.save_non_linear_terms = []
        if type(self.linear_correction_approach) is not str: raise ValueError(
            "\"linear_correction_approach\" must be a string")
        match self.linear_correction_approach.upper():
            case "NONE":
                ""
            case "CONSTANT":
                self.psi_parameters = nn.ParameterList([
                    nn.Parameter(torch.zeros(out_features, **self.factory_kwargs, requires_grad=False))
                    for out_features in layers[2:]
                ])
            case "TRAINABLE":
                self.psi_parameters = nn.ParameterList([
                    nn.Parameter(torch.zeros(out_features, **self.factory_kwargs))
                    for out_features in layers[2:]
                ])
            case "TRAINABLE+DECAY":
                self.psi_parameters = nn.ParameterList([
                    nn.Parameter(torch.zeros(out_features, **self.factory_kwargs))
                    for out_features in layers[2:]
                ])
            case "EXPECTATION":
                ""
            case _:
                raise ValueError(f"Invalid linear correction approach, \"{self.linear_correction_approach.upper()}\" choose from \"None\", \"Constant\", \"Trainable\", \"Trainable+Decay\", or \"Expectation\"")


    @property
    def architecture(self):
        return [self.weight_parameters[0].shape[1],]+[b.shape[0] for b in self.bias_parameters]

    def simple_initialiser(self, weight_init="xavier_normal"):

        with torch.no_grad():
            for weight in self.weight_parameters:
                if weight_init == "orthogonal":
                    nn.init.orthogonal_(weight)
                elif weight_init == "xavier_normal":
                    nn.init.xavier_normal_(weight)
                elif weight_init == "xavier_uniform":
                    nn.init.xavier_uniform_(weight)
                else:
                    raise ValueError("Unknown weight_init")

            for bias in self.bias_parameters:
                bias.zero_()

            if self.linear_correction_approach.upper() not in ["NONE", "EXPECTATION"]:
                for psi in self.psi_parameters:
                    psi.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the MLP network, given that tensor x is of shape [BATCH, 1, ..., n-1]
        :param x: Input data
        :return: Output data
        """

        if self.flatten: x = x.flatten(start_dim=1)

        if self.linear_correction_approach.upper()=="EXPECTATION": self.save_non_linear_terms = []
        if self.linear_correction_approach.upper() in ["NONE", "EXPECTATION"]:
            temporary = x
            for weight, bias, sigma in zip(self.weight_parameters[:-1], self.bias_parameters[:-1], self.activation):
                non_linear_term, linear_term = sigma(torch.einsum("ij, bj->bi", weight, temporary) + bias[None, :])
                if self.linear_correction_approach.upper()=="EXPECTATION": self.save_non_linear_terms.append(non_linear_term.detach().clone())
                temporary = linear_term * non_linear_term



            output = torch.einsum("ij, bj->bi", self.weight_parameters[-1], temporary) + self.bias_parameters[-1][None, :]
            if self.unflatten_shape is not None: output = output.unflatten(dim=1, sizes=self.unflatten_shape)
            return output

        else:
            temporary = x
            for i, (weight, bias, sigma) in enumerate(zip(self.weight_parameters[:-1], self.bias_parameters[:-1], self.activation)):
                if i == 0:
                    affine_term = torch.einsum("ij, bj->bi", weight, temporary) + bias[None, :]
                    non_linear_term, linear_term = sigma(affine_term)
                    temporary = linear_term * non_linear_term
                else:
                    affine_term = torch.einsum("ij, bj->bi", weight, temporary) + bias[None, :] + self.psi_parameters[i - 1][None, :] * non_linear_term
                    non_linear_term, linear_term = sigma(affine_term)
                    temporary = linear_term * non_linear_term


            output = torch.einsum("ij, bj->bi", self.weight_parameters[-1], temporary) + self.bias_parameters[-1][None, :] + self.psi_parameters[-1][None, :] * non_linear_term
            if self.unflatten_shape is not None: output = output.unflatten(dim=1, sizes=self.unflatten_shape)
            return output

    def singular_value_decompose(self, layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform singular value decomposition on layer specified. Then returns the SVD decomposed matrices for this layer.
        :param layer: Specified layer to perform SVD decomposition
        :return:
        """
        if type(layer) is not int: raise ValueError("\"layer\" must be an integer")
        if layer < 0 or layer >= len(self.architecture)-1: raise ValueError("\"layer\" must be between 0 and the number of layers")

        with torch.no_grad():
            # Access the weights for that corresponding layer
            W = self.weight_parameters[layer].detach()  # (m, n)
            # Perform singular value decomposition
            R, S, QT = torch.linalg.svd(W, full_matrices=True)
            L = torch.zeros_like(W)
            k = min(W.shape)
            L[:k, :k] = torch.diag(S)
        return R, L, QT


    def left_partial_diagonalise(self, layer : int, verbose: bool = False, output:bool = False, *, optimiser=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the left partial diagonalisation on layer specified. Then returns the SVD decomposed matrices for this layer.
        :param layer: Specify layer to left partial diagonalise
        :param verbose: Print details of the SVD decomposition
        :return: The SVD decomposed matrices for this layer.
        """
        if type(layer) is not int: raise ValueError("\"layer\" must be an integer")
        if layer < 0 or layer >= len(self.architecture)-2: raise ValueError("\"layer\" must be between 0 and the number of layers minus 1")

        # Perform singular value decomposition
        R, L, QT = self.singular_value_decompose(layer)

        if verbose: print(f"Layer {layer} SVD decomposition: R = {R.shape} L = {L.shape} QT = {QT.shape}")
        # Now a readjustment of surrounding layers, given the decomposition.
        with torch.no_grad():
            # Current Layer
            if verbose: print(f"Layer {layer}: W = {self.weight_parameters[layer].shape} b = {self.bias_parameters[layer].shape}", end="")
            self.weight_parameters[layer].copy_(L @ QT)
            self.bias_parameters[layer].copy_(R.T @ self.bias_parameters[layer])

            if hasattr(self, "psi_parameters") and layer != 0:
                if verbose: print(f" Psi = {self.psi_parameters[layer-1].shape}")
                self.psi_parameters[layer-1].copy_(R.T @ self.psi_parameters[layer-1])
            else:
                if verbose: print("")
            # Subsequent Layer
            if verbose: print(f"Layer {layer}: W = {self.weight_parameters[layer+1].shape} b = {self.bias_parameters[layer+1].shape}")
            self.weight_parameters[layer+1].copy_(self.weight_parameters[layer+1] @ R)

            # One needs to transform the optimiser's internal states, implemented for ADAM and MomentumGD. Please see diagonal approximation above
            if optimiser is not None:
                # Current layer: W -> W' = R.T @ W = L @ QT
                state = optimiser.state.get(self.weight_parameters[layer], None)
                if state is not None:
                    for key in ("exp_avg", "momentum_buffer"):
                        if key in state and self.transform_first_moment:
                            state[key].copy_(R.T @ state[key])
                    if "exp_avg_sq" in state and self.approx_transform_second_moment:
                        state["exp_avg_sq"].copy_((R.T ** 2) @ state["exp_avg_sq"])

                # Current layer bias: b -> b' = R.T @ b
                state = optimiser.state.get(self.bias_parameters[layer], None)
                if state is not None:
                    for key in ("exp_avg", "momentum_buffer"):
                        if key in state and self.transform_first_moment:
                            state[key].copy_(R.T @ state[key])
                    if "exp_avg_sq" in state and self.approx_transform_second_moment:
                        state["exp_avg_sq"].copy_((R.T ** 2) @ state["exp_avg_sq"])

                # Psi on previous connection: psi -> psi' = R.T @ psi
                if hasattr(self, "psi_parameters") and layer != 0:
                    state = optimiser.state.get(self.psi_parameters[layer - 1], None)
                    if state is not None:
                        for key in ("exp_avg", "momentum_buffer"):
                            if key in state and self.transform_first_moment:
                                state[key].copy_(R.T @ state[key])
                        if "exp_avg_sq" in state and self.approx_transform_second_moment:
                            state["exp_avg_sq"].copy_((R.T ** 2) @ state["exp_avg_sq"])

                # Subsequent layer: W_next -> W_next' = W_next @ R
                state = optimiser.state.get(self.weight_parameters[layer + 1], None)
                if state is not None:
                    for key in ("exp_avg", "momentum_buffer"):
                        if key in state and self.transform_first_moment:
                            state[key].copy_(state[key] @ R)
                    if "exp_avg_sq" in state and self.approx_transform_second_moment:
                        state["exp_avg_sq"].copy_(state["exp_avg_sq"] @ (R ** 2))

        if output: return R, L, QT


    def full_diagonalise(self, layer : int, verbose: bool = False, output:bool = False, *, optimiser=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the full diagonalisation on layer specified. Then returns the SVD decomposed matrices for this layer.
        :param layer: Specify layer to left partial diagonalise
        :param verbose: Print details of the SVD decomposition
        :return: The SVD decomposed matrices for this layer.
        """
        # Perform singular value decomposition
        if type(layer) is not int: raise ValueError("\"layer\" must be an integer")
        if layer < 1 or layer >= len(self.architecture)-2: raise ValueError("\"layer\" must be between 1 and the number of layers minus 1")

        R, L, QT = self.singular_value_decompose(layer)

        if verbose: print(f"Layer {layer} SVD decomposition: R = {R.shape} L = {L.shape} QT = {QT.shape}")

        # Now a readjustment of surrounding layers, given the decomposition.
        with torch.no_grad():
            # Current Layer
            if verbose: print(f"Layer {layer}: W2 = {self.weight_parameters[layer].shape} b2 = {self.bias_parameters[layer].shape}", end="")
            
            self.weight_parameters[layer].copy_(L)
            self.bias_parameters[layer].copy_(R.T @ self.bias_parameters[layer])

            if hasattr(self, "psi_parameters") and layer != 0:
                if verbose: print(f" Psi = {self.psi_parameters[layer-1].shape}")
                self.psi_parameters[layer-1].copy_(R.T @ self.psi_parameters[layer-1])
            else:
                if verbose: print("")
            
            # Prior Layer
            if verbose: print(f"Prior Layer: W1 = {self.weight_parameters[layer-1].shape} b1 = {self.bias_parameters[layer-1].shape}", end="")
            self.weight_parameters[layer-1].copy_(QT @ self.weight_parameters[layer-1])
            self.bias_parameters[layer-1].copy_(QT @ self.bias_parameters[layer-1])
            if hasattr(self, "psi_parameters") and layer > 1:
                if verbose: print(f" Psi = {self.psi_parameters[layer-2].shape}")
                self.psi_parameters[layer-2].copy_(QT @ self.psi_parameters[layer-2])
            else:
                if verbose: print("")


            # Subsequent Layer
            if verbose: print(f"Next Layer: W3 = {self.weight_parameters[layer+1].shape} b3 = {self.bias_parameters[layer+1].shape}")
            self.weight_parameters[layer+1].copy_(self.weight_parameters[layer+1] @ R)

            if optimiser is not None:
                # Current layer weight: W2 -> W2' = R.T @ W2 @ QT.T = L
                state = optimiser.state.get(self.weight_parameters[layer], None)
                if state is not None:
                    for key in ("exp_avg", "momentum_buffer"):
                        if key in state and self.transform_first_moment:
                            state[key].copy_(R.T @ state[key] @ QT.T)
                    if "exp_avg_sq" in state and self.approx_transform_second_moment:
                        state["exp_avg_sq"].copy_((R.T ** 2) @ state["exp_avg_sq"] @ (QT.T ** 2))

                # Current layer bias: b2 -> b2' = R.T @ b2
                state = optimiser.state.get(self.bias_parameters[layer], None)
                if state is not None:
                    for key in ("exp_avg", "momentum_buffer"):
                        if key in state and self.transform_first_moment:
                            state[key].copy_(R.T @ state[key])
                    if "exp_avg_sq" in state and self.approx_transform_second_moment:
                        state["exp_avg_sq"].copy_((R.T ** 2) @ state["exp_avg_sq"])

                # Psi on previous connection: psi_{layer-1} -> R.T @ psi_{layer-1}
                if hasattr(self, "psi_parameters") and layer != 0:
                    state = optimiser.state.get(self.psi_parameters[layer - 1], None)
                    if state is not None:
                        for key in ("exp_avg", "momentum_buffer"):
                            if key in state and self.transform_first_moment:
                                state[key].copy_(R.T @ state[key])
                        if "exp_avg_sq" in state and self.approx_transform_second_moment:
                            state["exp_avg_sq"].copy_((R.T ** 2) @ state["exp_avg_sq"])

                # Prior layer weight: W1 -> W1' = QT @ W1
                state = optimiser.state.get(self.weight_parameters[layer - 1], None)
                if state is not None:
                    for key in ("exp_avg", "momentum_buffer"):
                        if key in state and self.transform_first_moment:
                            state[key].copy_(QT @ state[key])
                    if "exp_avg_sq" in state and self.approx_transform_second_moment:
                        state["exp_avg_sq"].copy_((QT ** 2) @ state["exp_avg_sq"])

                # Prior layer bias: b1 -> b1' = QT @ b1
                state = optimiser.state.get(self.bias_parameters[layer - 1], None)
                if state is not None:
                    for key in ("exp_avg", "momentum_buffer"):
                        if key in state and self.transform_first_moment:
                            state[key].copy_(QT @ state[key])
                    if "exp_avg_sq" in state and self.approx_transform_second_moment:
                        state["exp_avg_sq"].copy_((QT ** 2) @ state["exp_avg_sq"])

                # Psi on earlier connection: psi_{layer-2} -> QT @ psi_{layer-2}
                if hasattr(self, "psi_parameters") and layer > 1:
                    state = optimiser.state.get(self.psi_parameters[layer - 2], None)
                    if state is not None:
                        for key in ("exp_avg", "momentum_buffer"):
                            if key in state and self.transform_first_moment:
                                state[key].copy_(QT @ state[key])
                        if "exp_avg_sq" in state and self.approx_transform_second_moment:
                            state["exp_avg_sq"].copy_((QT ** 2) @ state["exp_avg_sq"])

                # Subsequent layer weight: W3 -> W3' = W3 @ R
                state = optimiser.state.get(self.weight_parameters[layer + 1], None)
                if state is not None:
                    for key in ("exp_avg", "momentum_buffer"):
                        if key in state and self.transform_first_moment:
                            state[key].copy_(state[key] @ R)
                    if "exp_avg_sq" in state and self.approx_transform_second_moment:
                        state["exp_avg_sq"].copy_(state["exp_avg_sq"] @ (R ** 2))

        if output: return R, L, QT

    def get_singular_values(self, *, layer : int, use_torch:bool = False) -> list:
        """
        Retrieves the singular values for the specified layer
        :param layer: Specify layer to get singular values
        :param use_torch: Return singular values as a torch tensor
        :return: List of singular values
        """

        with torch.no_grad():
            _, S, _ = self.singular_value_decompose(layer)
        if use_torch: return torch.diag(S)
        else: return torch.diag(S).tolist()


    def neurogenerate(self, layer : int, verbose: bool = False, *, optimiser=None):
        """
        Adds a new scaffold neuron to the specified layer, uses zero-initialised for new entries (unbiased spontaneous symmetry breaking).
        :param layer: Specify layer R^n\rightarrowR^m to add scaffold neuron to form R^n\rightarrowR^{m+1}
        :param verbose: Print details
        """
        if type(layer) is not int: raise ValueError("\"layer\" must be an integer")
        if layer < 0 or layer >= len(self.architecture)-2: raise ValueError("\"layer\" must be between 0 and the number of layers minus 1")

        # Perform singular value decomposition
        R, L, QT = self.singular_value_decompose(layer)

        if verbose: print(f"Layer {layer} Neurogenesis: Going from R^{L.shape[1]}→R^{L.shape[0]} to R^{L.shape[1]}→R^{L.shape[0]+1}")

        with torch.no_grad():
            if optimiser is not None:
                old_weight_parameter = self.weight_parameters[layer]
                old_bias_parameter = self.bias_parameters[layer]
                old_next_weight_parameter = self.weight_parameters[layer + 1]
                if hasattr(self, "psi_parameters") and layer != 0:
                    old_psi_parameter = self.psi_parameters[layer - 1]

            if verbose: print(f"W1 = {self.weight_parameters[layer].shape}, b1 = {self.bias_parameters[layer].shape}, W2 = {self.weight_parameters[layer + 1].shape}")
            self.weight_parameters[layer] = nn.Parameter(torch.vstack([L, torch.zeros_like(L[0, :])])@QT)
            self.bias_parameters[layer] = nn.Parameter(torch.hstack([R.T@self.bias_parameters[layer], torch.zeros_like(self.bias_parameters[layer][0])[None]]))
            self.weight_parameters[layer+1] = nn.Parameter(torch.hstack([self.weight_parameters[layer+1]@R, 0.01*torch.randn_like(self.weight_parameters[layer+1][:, 0][:, None])])) #zeros_like
            
            if hasattr(self, "psi_parameters") and layer != 0:
                if verbose: print(f" Psi = {self.psi_parameters[layer-1].shape}")
                self.psi_parameters[layer-1] = nn.Parameter(torch.hstack([R.T@self.psi_parameters[layer-1], torch.zeros_like(self.psi_parameters[layer-1][0])[None]]), requires_grad=False if self.linear_correction_approach.upper() in ["CONSTANT"] else True)
            else:
                if verbose: print("")

            if optimiser is not None:
                # This should purposefully crash non ADAM/SGD optimisers, as they'll be missing states. These need manually solving their transforms and implementing into this part.
                # DO NOT TRY AND RUN NEW OPTIMISERS WITHOUT SUCH TRANSFORMS
                state = optimiser.state.pop(old_weight_parameter, None)
                if state is not None:
                    new_state = {}
                    if "step" in state:
                        new_state["step"] = state["step"]
                    for key in ("exp_avg", "momentum_buffer"):
                        if key in state and self.transform_first_moment:
                            new_state[key] = torch.vstack([R.T @ state[key], torch.zeros_like(state[key][0:1, :])])
                    if "exp_avg_sq" in state and self.approx_transform_second_moment:
                        new_state["exp_avg_sq"] = torch.vstack([
                            (R.T ** 2) @ state["exp_avg_sq"],
                            torch.zeros_like(state["exp_avg_sq"][0:1, :])
                        ])
                    optimiser.state[self.weight_parameters[layer]] = new_state

                state = optimiser.state.pop(old_bias_parameter, None)
                if state is not None:
                    new_state = {}
                    if "step" in state:
                        new_state["step"] = state["step"]
                    for key in ("exp_avg", "momentum_buffer"):
                        if key in state and self.transform_first_moment:
                            new_state[key] = torch.hstack([R.T @ state[key], torch.zeros_like(state[key][0:1])])
                    if "exp_avg_sq" in state and self.approx_transform_second_moment:
                        new_state["exp_avg_sq"] = torch.hstack([
                            (R.T ** 2) @ state["exp_avg_sq"],
                            torch.zeros_like(state["exp_avg_sq"][0:1])
                        ])
                    optimiser.state[self.bias_parameters[layer]] = new_state

                state = optimiser.state.pop(old_next_weight_parameter, None)
                if state is not None:
                    new_state = {}
                    if "step" in state:
                        new_state["step"] = state["step"]
                    for key in ("exp_avg", "momentum_buffer"):
                        if key in state and self.transform_first_moment:
                            new_state[key] = torch.hstack([state[key] @ R, torch.zeros_like(state[key][:, 0:1])])
                    if "exp_avg_sq" in state and self.approx_transform_second_moment:
                        new_state["exp_avg_sq"] = torch.hstack([
                            state["exp_avg_sq"] @ (R ** 2),
                            torch.zeros_like(state["exp_avg_sq"][:, 0:1])
                        ])
                    optimiser.state[self.weight_parameters[layer + 1]] = new_state

                if hasattr(self, "psi_parameters") and layer != 0:
                    state = optimiser.state.pop(old_psi_parameter, None)
                    if state is not None:
                        new_state = {}
                        if "step" in state:
                            new_state["step"] = state["step"]
                        for key in ("exp_avg", "momentum_buffer"):
                            if key in state and self.transform_first_moment:
                                new_state[key] = torch.hstack([R.T @ state[key], torch.zeros_like(state[key][0:1])])
                        if "exp_avg_sq" in state and self.approx_transform_second_moment:
                            new_state["exp_avg_sq"] = torch.hstack([
                                (R.T ** 2) @ state["exp_avg_sq"],
                                torch.zeros_like(state["exp_avg_sq"][0:1])
                            ])
                        optimiser.state[self.psi_parameters[layer - 1]] = new_state

                for group in optimiser.param_groups:
                    for i, parameter in enumerate(group["params"]):
                        if parameter is old_weight_parameter:
                            group["params"][i] = self.weight_parameters[layer]
                        elif parameter is old_bias_parameter:
                            group["params"][i] = self.bias_parameters[layer]
                        elif parameter is old_next_weight_parameter:
                            group["params"][i] = self.weight_parameters[layer + 1]
                        elif hasattr(self, "psi_parameters") and layer != 0 and parameter is old_psi_parameter:
                            group["params"][i] = self.psi_parameters[layer - 1]
            
            if verbose: print(f"W1 = {self.weight_parameters[layer].shape}, b1 = {self.bias_parameters[layer].shape}, W2 = {self.weight_parameters[layer+1].shape}")


            

    def neurodegenerate(self, layer : int, verbose: bool = False, *, optimiser=None):
        """
        Removes smallest singular value diagonalised neuron from the specified layer.
        :param layer: Specify layer R^n\rightarrowR^m to add scaffold neuron to form R^n\rightarrowR^{m-1}
        :param verbose: Print details
        """
        if type(layer) is not int: raise ValueError("\"layer\" must be an integer")
        if layer < 0 or layer >= len(self.architecture)-2: raise ValueError("\"layer\" must be between 0 and the number of layers minus 1")

        # Perform singular value decomposition
        R, L, QT = self.singular_value_decompose(layer)
        if verbose: print(f"Layer {layer} Neurodegenerate: Going from R^{L.shape[1]}→R^{L.shape[0]} to R^{L.shape[1]}→R^{L.shape[0]-1}")
        if verbose: print(f"Singular value of magnitude {torch.diag(L)[-1]:.3E} removed!")

        # Perform singular value decomposition
        R, L, QT = self.singular_value_decompose(layer)

        # First perform effectively the left-partial diagonalisation, and assume singular values are ordered such that last entry is always smallest.
        with torch.no_grad():
            if optimiser is not None:
                old_weight_parameter = self.weight_parameters[layer]
                old_bias_parameter = self.bias_parameters[layer]
                old_next_weight_parameter = self.weight_parameters[layer + 1]
                if hasattr(self, "psi_parameters") and layer != 0:
                    old_psi_parameter = self.psi_parameters[layer - 1]
            # Current Layer
            if verbose: print(f"Layer {layer}: W = {self.weight_parameters[layer].shape} b = {self.bias_parameters[layer].shape}", end="")
            self.weight_parameters[layer] = nn.Parameter(L[:-1, :] @ QT)
            b_star = (R.T @ self.bias_parameters[layer]).clone()[-1]
            self.bias_parameters[layer]= nn.Parameter((R.T @ self.bias_parameters[layer])[:-1])

            # This is assumed to have already relaxed to zero, so entry is deleted without concern
            if hasattr(self, "psi_parameters") and layer != 0:
                if verbose: print(f" Psi = {self.psi_parameters[layer - 1].shape}")
                self.psi_parameters[layer - 1] = nn.Parameter((R.T @ self.psi_parameters[layer - 1])[:-1], requires_grad=False if self.linear_correction_approach.upper() in ["CONSTANT"] else True)
            else:
                if verbose: print("")

            # Subsequent Layer
            if verbose: print(f"Layer {layer}: W = {self.weight_parameters[layer + 1].shape} b = {self.bias_parameters[layer + 1].shape}")
            W_star = (self.weight_parameters[layer + 1] @ R).clone()[:, -1]
            self.weight_parameters[layer + 1] = nn.Parameter((self.weight_parameters[layer + 1] @ R).clone()[:, :-1])

            # Now update o
            if self.intrinsic_length_approach.upper() in ["CONSTANT", "TRAINABLE"]: self.activation[layer].o = self.activation[layer].o + b_star ** 2

            # Now update subsequent bias or psi
            match self.linear_correction_approach.upper():
                case "CONSTANT":
                    self.psi_parameters[layer].add_(W_star * b_star)
                case "TRAINABLE":
                    self.psi_parameters[layer].add_(W_star * b_star)
                case "TRAINABLE+DECAY":
                    self.psi_parameters[layer].add_(W_star * b_star)
                case "EXPECTATION":
                    factor = self.save_non_linear_terms[layer].mean() * W_star * b_star
                    self.bias_parameters[layer+1].add_(factor)

            if optimiser is not None:
                # This should purposefully crash non ADAM/SGD optimisers, as they'll be missing states. These need manually solving their transforms and implementing into this part.
                # DO NOT TRY AND RUN NEW OPTIMISERS WITHOUT SUCH TRANSFORMS

                # Similarly o and psi additive transforms are not performed on the optimiser states, as they are non-linear transforms of their states, and very unclear how they should effectively update.
                # This is especially true for the squared states.
                state = optimiser.state.pop(old_weight_parameter, None)
                if state is not None:
                    new_state = {}
                    if "step" in state:
                        new_state["step"] = state["step"]
                    for key in ("exp_avg", "momentum_buffer"):
                        if key in state and self.transform_first_moment:
                            new_state[key] = (R.T @ state[key])[:-1, :]
                    if "exp_avg_sq" in state and self.approx_transform_second_moment:
                        new_state["exp_avg_sq"] = ((R.T ** 2) @ state["exp_avg_sq"])[:-1, :]
                    optimiser.state[self.weight_parameters[layer]] = new_state

                state = optimiser.state.pop(old_bias_parameter, None)
                if state is not None:
                    new_state = {}
                    if "step" in state:
                        new_state["step"] = state["step"]
                    for key in ("exp_avg", "momentum_buffer"):
                        if key in state and self.transform_first_moment:
                            new_state[key] = (R.T @ state[key])[:-1]
                    if "exp_avg_sq" in state and self.approx_transform_second_moment:
                        new_state["exp_avg_sq"] = ((R.T ** 2) @ state["exp_avg_sq"])[:-1]
                    optimiser.state[self.bias_parameters[layer]] = new_state

                state = optimiser.state.pop(old_next_weight_parameter, None)
                if state is not None:
                    new_state = {}
                    if "step" in state:
                        new_state["step"] = state["step"]
                    for key in ("exp_avg", "momentum_buffer"):
                        if key in state and self.transform_first_moment:
                            new_state[key] = (state[key] @ R)[:, :-1]
                    if "exp_avg_sq" in state and self.approx_transform_second_moment:
                        new_state["exp_avg_sq"] = (state["exp_avg_sq"] @ (R ** 2))[:, :-1]
                    optimiser.state[self.weight_parameters[layer + 1]] = new_state

                if hasattr(self, "psi_parameters") and layer != 0:
                    state = optimiser.state.pop(old_psi_parameter, None)
                    if state is not None:
                        new_state = {}
                        if "step" in state:
                            new_state["step"] = state["step"]
                        for key in ("exp_avg", "momentum_buffer"):
                            if key in state and self.transform_first_moment:
                                new_state[key] = (R.T @ state[key])[:-1]
                        if "exp_avg_sq" in state and self.approx_transform_second_moment:
                            new_state["exp_avg_sq"] = ((R.T ** 2) @ state["exp_avg_sq"])[:-1]
                        optimiser.state[self.psi_parameters[layer - 1]] = new_state

                for group in optimiser.param_groups:
                    for i, parameter in enumerate(group["params"]):
                        if parameter is old_weight_parameter:
                            group["params"][i] = self.weight_parameters[layer]
                        elif parameter is old_bias_parameter:
                            group["params"][i] = self.bias_parameters[layer]
                        elif parameter is old_next_weight_parameter:
                            group["params"][i] = self.weight_parameters[layer + 1]
                        elif hasattr(self, "psi_parameters") and layer != 0 and parameter is old_psi_parameter:
                            group["params"][i] = self.psi_parameters[layer - 1]

    def auto_neuroadapation(self, layer: int, scaffold_neuron_threshold: int=2, singular_value_threshold: float = 0.1, verbose: bool = False, *, optimiser=None):
        """
        Determines whether to add scaffold neurons or remove neurons based on the singular values of the layer.
        :param layer: Specified layer
        :param scaffold_neuron_threshold: The number of scaffold neurons to always maintain below threshold
        :param singular_value_threshold: The threshold for those singular values
        :param verbose: Print details
        """
        if type(layer) is not int: raise ValueError("\"layer\" must be an integer")
        if layer < 0 or layer >= len(self.architecture)-2: raise ValueError("\"layer\" must be between 0 and the number of layers minus 1")
        if type(scaffold_neuron_threshold) is not int: raise ValueError("\"scaffold_neuron_threshold\" must be an integer")
        if type(singular_value_threshold) is not float: raise ValueError("\"singular_value_threshold\" must be a float")
        if scaffold_neuron_threshold<=0: raise ValueError("\"scaffold_neuron_threshold\" must be greater than 0")
        if singular_value_threshold<=0: raise ValueError("\"singular_value_threshold\" must be greater than 0")
        if type(verbose) is not bool: raise ValueError("\"verbose\" must be a boolean")

        singular_values = self.get_singular_values(layer=layer, use_torch=True)
        if verbose: print(f"Singular values for layer {layer}: {singular_values.tolist()}")

        active_scaffold_neurons = torch.count_nonzero(singular_values <= singular_value_threshold)

        if verbose: print(f"Active scaffold neurons: {active_scaffold_neurons} below {singular_value_threshold}")
        scaffold_neuron_difference = int(scaffold_neuron_threshold - active_scaffold_neurons)
        if scaffold_neuron_difference > 0:
            if verbose: print(f"Adding {scaffold_neuron_difference} scaffold neurons to layer {layer}")
            for i in range(scaffold_neuron_difference): self.neurogenerate(layer=layer, optimiser=optimiser)
        elif scaffold_neuron_difference < 0:
            if verbose: print(f"Removing {-scaffold_neuron_difference} scaffold neurons to layer {layer}")
            for i in range(-scaffold_neuron_difference): self.neurodegenerate(layer=layer, optimiser=optimiser)

    def add_first_layer_neurons(self, number_of_neurons: int, verbose: bool = False, *, optimiser=None):
        """
        Adds new first-layer input coordinates by appending zero-initialised columns to the first weight matrix.
        :param number_of_neurons: Number of input coordinates to append
        :param verbose: Print details
        """
        if type(number_of_neurons) is not int: raise ValueError("\"number_of_neurons\" must be an integer")
        if number_of_neurons <= 0: raise ValueError("\"number_of_neurons\" must be greater than 0")

        with torch.no_grad():
            if optimiser is not None:
                old_weight_parameter = self.weight_parameters[0]

            if verbose: print(
                f"First-layer input expansion: {self.weight_parameters[0].shape[1]} -> {self.weight_parameters[0].shape[1] + number_of_neurons}")

            self.weight_parameters[0] = nn.Parameter(
                torch.hstack([
                    self.weight_parameters[0],
                    torch.zeros(
                        (self.weight_parameters[0].shape[0], number_of_neurons),
                        device=self.weight_parameters[0].device,
                        dtype=self.weight_parameters[0].dtype,
                    ),
                ])
            )

            if optimiser is not None:
                state = optimiser.state.pop(old_weight_parameter, None)
                if state is not None:
                    new_state = {}
                    if "step" in state:
                        new_state["step"] = state["step"]
                    for key in ("exp_avg", "momentum_buffer"):
                        if key in state and self.transform_first_moment:
                            new_state[key] = torch.hstack([
                                state[key],
                                torch.zeros(
                                    (state[key].shape[0], number_of_neurons),
                                    device=state[key].device,
                                    dtype=state[key].dtype,
                                ),
                            ])
                    if "exp_avg_sq" in state and self.approx_transform_second_moment:
                        new_state["exp_avg_sq"] = torch.hstack([
                            state["exp_avg_sq"],
                            torch.zeros(
                                (state["exp_avg_sq"].shape[0], number_of_neurons),
                                device=state["exp_avg_sq"].device,
                                dtype=state["exp_avg_sq"].dtype,
                            ),
                        ])
                    optimiser.state[self.weight_parameters[0]] = new_state

                for group in optimiser.param_groups:
                    for i, parameter in enumerate(group["params"]):
                        if parameter is old_weight_parameter:
                            group["params"][i] = self.weight_parameters[0]

    def remove_first_layer_neurons(self, number_of_neurons: int, verbose: bool = False, *, optimiser=None):
        """
        Removes last first-layer input coordinates by deleting the final columns of the first weight matrix.
        :param number_of_neurons: Number of input coordinates to remove
        :param verbose: Print details
        """
        if type(number_of_neurons) is not int: raise ValueError("\"number_of_neurons\" must be an integer")
        if number_of_neurons <= 0: raise ValueError("\"number_of_neurons\" must be greater than 0")
        if number_of_neurons >= self.weight_parameters[0].shape[1]:
            raise ValueError("Cannot remove all first-layer input coordinates")

        with torch.no_grad():
            if optimiser is not None:
                old_weight_parameter = self.weight_parameters[0]

            if verbose: print(
                f"First-layer input reduction: {self.weight_parameters[0].shape[1]} -> {self.weight_parameters[0].shape[1] - number_of_neurons}")

            self.weight_parameters[0] = nn.Parameter(self.weight_parameters[0][:, :-number_of_neurons])

            if optimiser is not None:
                state = optimiser.state.pop(old_weight_parameter, None)
                if state is not None:
                    new_state = {}
                    if "step" in state:
                        new_state["step"] = state["step"]
                    for key in ("exp_avg", "momentum_buffer"):
                        if key in state and self.transform_first_moment:
                            new_state[key] = state[key][:, :-number_of_neurons]
                    if "exp_avg_sq" in state and self.approx_transform_second_moment:
                        new_state["exp_avg_sq"] = state["exp_avg_sq"][:, :-number_of_neurons]
                    optimiser.state[self.weight_parameters[0]] = new_state

                for group in optimiser.param_groups:
                    for i, parameter in enumerate(group["params"]):
                        if parameter is old_weight_parameter:
                            group["params"][i] = self.weight_parameters[0]

    def add_last_layer_neurons(self, number_of_neurons: int, verbose: bool = False, *, optimiser=None):
        """
        Adds new last-layer output neurons using zero initialisation.
        :param number_of_neurons: Number of output neurons to append
        :param verbose: Print details
        """
        if type(number_of_neurons) is not int: raise ValueError("\"number_of_neurons\" must be an integer")
        if number_of_neurons <= 0: raise ValueError("\"number_of_neurons\" must be greater than 0")

        with torch.no_grad():
            if optimiser is not None:
                old_weight_parameter = self.weight_parameters[-1]
                old_bias_parameter = self.bias_parameters[-1]
                if hasattr(self, "psi_parameters"):
                    old_psi_parameter = self.psi_parameters[-1]

            if verbose: print(
                f"Last-layer output expansion: {self.weight_parameters[-1].shape[0]} -> {self.weight_parameters[-1].shape[0] + number_of_neurons}")

            self.weight_parameters[-1] = nn.Parameter(
                torch.vstack([
                    self.weight_parameters[-1],
                    torch.zeros(
                        (number_of_neurons, self.weight_parameters[-1].shape[1]),
                        device=self.weight_parameters[-1].device,
                        dtype=self.weight_parameters[-1].dtype,
                    ),
                ])
            )
            self.bias_parameters[-1] = nn.Parameter(
                torch.hstack([
                    self.bias_parameters[-1],
                    torch.zeros(
                        number_of_neurons,
                        device=self.bias_parameters[-1].device,
                        dtype=self.bias_parameters[-1].dtype,
                    ),
                ])
            )

            if hasattr(self, "psi_parameters"):
                self.psi_parameters[-1] = nn.Parameter(
                    torch.hstack([
                        self.psi_parameters[-1],
                        torch.zeros(
                            number_of_neurons,
                            device=self.psi_parameters[-1].device,
                            dtype=self.psi_parameters[-1].dtype,
                        ),
                    ]),
                    requires_grad=False if self.linear_correction_approach.upper() in ["CONSTANT"] else True
                )

            if optimiser is not None:
                state = optimiser.state.pop(old_weight_parameter, None)
                if state is not None:
                    new_state = {}
                    if "step" in state:
                        new_state["step"] = state["step"]
                    for key in ("exp_avg", "momentum_buffer"):
                        if key in state and self.transform_first_moment:
                            new_state[key] = torch.vstack([
                                state[key],
                                torch.zeros(
                                    (number_of_neurons, state[key].shape[1]),
                                    device=state[key].device,
                                    dtype=state[key].dtype,
                                ),
                            ])
                    if "exp_avg_sq" in state and self.approx_transform_second_moment:
                        new_state["exp_avg_sq"] = torch.vstack([
                            state["exp_avg_sq"],
                            torch.zeros(
                                (number_of_neurons, state["exp_avg_sq"].shape[1]),
                                device=state["exp_avg_sq"].device,
                                dtype=state["exp_avg_sq"].dtype,
                            ),
                        ])
                    optimiser.state[self.weight_parameters[-1]] = new_state

                state = optimiser.state.pop(old_bias_parameter, None)
                if state is not None:
                    new_state = {}
                    if "step" in state:
                        new_state["step"] = state["step"]
                    for key in ("exp_avg", "momentum_buffer"):
                        if key in state and self.transform_first_moment:
                            new_state[key] = torch.hstack([
                                state[key],
                                torch.zeros(number_of_neurons, device=state[key].device, dtype=state[key].dtype),
                            ])
                    if "exp_avg_sq" in state and self.approx_transform_second_moment:
                        new_state["exp_avg_sq"] = torch.hstack([
                            state["exp_avg_sq"],
                            torch.zeros(number_of_neurons, device=state["exp_avg_sq"].device,
                                        dtype=state["exp_avg_sq"].dtype),
                        ])
                    optimiser.state[self.bias_parameters[-1]] = new_state

                if hasattr(self, "psi_parameters"):
                    state = optimiser.state.pop(old_psi_parameter, None)
                    if state is not None:
                        new_state = {}
                        if "step" in state:
                            new_state["step"] = state["step"]
                        for key in ("exp_avg", "momentum_buffer"):
                            if key in state and self.transform_first_moment:
                                new_state[key] = torch.hstack([
                                    state[key],
                                    torch.zeros(number_of_neurons, device=state[key].device, dtype=state[key].dtype),
                                ])
                        if "exp_avg_sq" in state and self.approx_transform_second_moment:
                            new_state["exp_avg_sq"] = torch.hstack([
                                state["exp_avg_sq"],
                                torch.zeros(number_of_neurons, device=state["exp_avg_sq"].device,
                                            dtype=state["exp_avg_sq"].dtype),
                            ])
                        optimiser.state[self.psi_parameters[-1]] = new_state

                for group in optimiser.param_groups:
                    for i, parameter in enumerate(group["params"]):
                        if parameter is old_weight_parameter:
                            group["params"][i] = self.weight_parameters[-1]
                        elif parameter is old_bias_parameter:
                            group["params"][i] = self.bias_parameters[-1]
                        elif hasattr(self, "psi_parameters") and parameter is old_psi_parameter:
                            group["params"][i] = self.psi_parameters[-1]

    def remove_last_layer_neurons(self, number_of_neurons: int, verbose: bool = False, *, optimiser=None):
        """
        Removes last last-layer output neurons by deleting the final rows/entries.
        :param number_of_neurons: Number of output neurons to remove
        :param verbose: Print details
        """
        if type(number_of_neurons) is not int: raise ValueError("\"number_of_neurons\" must be an integer")
        if number_of_neurons <= 0: raise ValueError("\"number_of_neurons\" must be greater than 0")
        if number_of_neurons >= self.weight_parameters[-1].shape[0]:
            raise ValueError("Cannot remove all last-layer output neurons")

        with torch.no_grad():
            if optimiser is not None:
                old_weight_parameter = self.weight_parameters[-1]
                old_bias_parameter = self.bias_parameters[-1]
                if hasattr(self, "psi_parameters"):
                    old_psi_parameter = self.psi_parameters[-1]

            if verbose: print(
                f"Last-layer output reduction: {self.weight_parameters[-1].shape[0]} -> {self.weight_parameters[-1].shape[0] - number_of_neurons}")

            self.weight_parameters[-1] = nn.Parameter(self.weight_parameters[-1][:-number_of_neurons, :])
            self.bias_parameters[-1] = nn.Parameter(self.bias_parameters[-1][:-number_of_neurons])

            if hasattr(self, "psi_parameters"):
                self.psi_parameters[-1] = nn.Parameter(
                    self.psi_parameters[-1][:-number_of_neurons],
                    requires_grad=False if self.linear_correction_approach.upper() in ["CONSTANT"] else True
                )

            if optimiser is not None:
                state = optimiser.state.pop(old_weight_parameter, None)
                if state is not None:
                    new_state = {}
                    if "step" in state:
                        new_state["step"] = state["step"]
                    for key in ("exp_avg", "momentum_buffer"):
                        if key in state and self.transform_first_moment:
                            new_state[key] = state[key][:-number_of_neurons, :]
                    if "exp_avg_sq" in state and self.approx_transform_second_moment:
                        new_state["exp_avg_sq"] = state["exp_avg_sq"][:-number_of_neurons, :]
                    optimiser.state[self.weight_parameters[-1]] = new_state

                state = optimiser.state.pop(old_bias_parameter, None)
                if state is not None:
                    new_state = {}
                    if "step" in state:
                        new_state["step"] = state["step"]
                    for key in ("exp_avg", "momentum_buffer"):
                        if key in state and self.transform_first_moment:
                            new_state[key] = state[key][:-number_of_neurons]
                    if "exp_avg_sq" in state and self.approx_transform_second_moment:
                        new_state["exp_avg_sq"] = state["exp_avg_sq"][:-number_of_neurons]
                    optimiser.state[self.bias_parameters[-1]] = new_state

                if hasattr(self, "psi_parameters"):
                    state = optimiser.state.pop(old_psi_parameter, None)
                    if state is not None:
                        new_state = {}
                        if "step" in state:
                            new_state["step"] = state["step"]
                        for key in ("exp_avg", "momentum_buffer"):
                            if key in state and self.transform_first_moment:
                                new_state[key] = state[key][:-number_of_neurons]
                        if "exp_avg_sq" in state and self.approx_transform_second_moment:
                            new_state["exp_avg_sq"] = state["exp_avg_sq"][:-number_of_neurons]
                        optimiser.state[self.psi_parameters[-1]] = new_state

                for group in optimiser.param_groups:
                    for i, parameter in enumerate(group["params"]):
                        if parameter is old_weight_parameter:
                            group["params"][i] = self.weight_parameters[-1]
                        elif parameter is old_bias_parameter:
                            group["params"][i] = self.bias_parameters[-1]
                        elif hasattr(self, "psi_parameters") and parameter is old_psi_parameter:
                            group["params"][i] = self.psi_parameters[-1]