import jax
import jax.numpy as jnp

# Normalization Layers
from typing import (Any, Callable, Iterable, Optional, Tuple, Union)

PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?

Axes = Union[int, Iterable[int]]

def _canonicalize_axes(rank: int, axes: Axes) -> Tuple[int, ...]:
    """Returns a tuple of deduplicated, sorted, and positive axes."""
    if not isinstance(axes, Iterable):
        axes = (axes,)
    return tuple(set([rank + axis if axis < 0 else axis for axis in axes]))


def _abs_sq(x):
    """Computes the elementwise square of the absolute value |x|^2."""
    if jnp.iscomplexobj(x):
        return jax.lax.square(jax.lax.real(x)) + jax.lax.square(jax.lax.imag(x))
    else:
        return jax.lax.square(x)

def _compute_stats(x: Array, axes: Axes,
                   dtype: Optional[Dtype],
                   axis_name: Optional[str] = None,
                   axis_index_groups: Any = None):
    """Computes mean and variance statistics.
    Returns:
      A pair ``(mean, var)``.
    """
    if dtype is None:
      dtype = jnp.result_type(x)
    # promote x to at least float32, this avoids half precision computation
    # but preserves double or complex floating points
    dtype = jnp.promote_types(dtype, jnp.float32)
    x = jnp.asarray(x, dtype)

    mean = jnp.mean(x, axes)
    mean2 = jnp.mean(_abs_sq(x), axes)
    if axis_name is not None:
      concatenated_mean = jnp.concatenate([mean, mean2])
      mean, mean2 = jnp.split(
          jax.lax.pmean(
              concatenated_mean,
              axis_name=axis_name,
              axis_index_groups=axis_index_groups), 2)
    # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
    # to floating point round-off errors.
    var = jnp.maximum(0., mean2 - _abs_sq(mean))
    return mean, var


def _normalize(x: Array, mean: Array, var: Array,
               reduction_axes: Axes, feature_axes: Axes,
               dtype: Dtype, param_dtype: Dtype,
               epsilon: float,
               bias: Array,
               scale: Array):
    """"Normalizes the input of a normalization layer and optionally applies a learned scale and bias.

    Arguments:
    mdl: Module to apply the normalization in (normalization params will reside
        in this module).
    x: The input.
    mean: Mean to use for normalization.
    var: Variance to use for normalization.
    reduction_axes: The axes in ``x`` to reduce.
    feature_axes: Axes containing features. A separate bias and scale is learned
        for each specified feature.
    dtype: The dtype of the result (default: infer from input and params).
    param_dtype: The dtype of the parameters.
    epsilon: Normalization epsilon.
    use_bias: If true, add a bias term to the output.
    use_scale: If true, scale the output.
    bias_init: Initialization function for the bias term.
    scale_init: Initialization function for the scaling function.

    Returns:
    The normalized input.
    """
    reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
    feature_axes = _canonicalize_axes(x.ndim, feature_axes)
    stats_shape = list(x.shape)
    for axis in reduction_axes:
        stats_shape[axis] = 1
    mean = mean.reshape(stats_shape)
    var = var.reshape(stats_shape)
    feature_shape = [1] * x.ndim
    reduced_feature_shape = []
    for ax in feature_axes:
        feature_shape[ax] = x.shape[ax]
        reduced_feature_shape.append(x.shape[ax])
    y = x - mean
    mul = jax.lax.rsqrt(var + epsilon)
    args = [x]
    # scale = mdl.param('scale', scale_init, reduced_feature_shape,
    #                     param_dtype).reshape(feature_shape)
    mul *= scale
    args.append(scale)
    y *= mul
    # bias = mdl.param('bias', bias_init, reduced_feature_shape,
    #                     param_dtype).reshape(feature_shape)
    y += bias
    args.append(bias)
    #   dtype = canonicalize_dtype(*args, dtype=dtype)
    return jnp.asarray(y)


def LayerNorm( x, epsilon: float = 1e-6, dtype: Optional[Dtype] = None, param_dtype: Dtype = jnp.float32, use_bias: bool = True,use_scale: bool = True,
                bias: Array = jax.nn.initializers.zeros, scale: Array = jax.nn.initializers.ones, reduction_axes: Axes = -1,feature_axes: Axes = -1,
                axis_name: Optional[str] = None, axis_index_groups: Any = None ):
    """Layer normalization (look at the _normalization function to see the role of the arguments) """
    mean, var = _compute_stats(x, reduction_axes, dtype, axis_name, axis_index_groups)
    return _normalize( x, mean, var, reduction_axes, feature_axes, dtype, param_dtype, epsilon, bias, scale)


def BatchNorm( x, axis: int=-1, momentum:float=0.05, epsilon: float = 1e-5, dtype: Optional[Dtype] = None,
                param_dtype: Dtype = jnp.float32,
                bias: Array = jax.nn.initializers.zeros, scale: Array = jax.nn.initializers.zeros,
                axis_name: Optional[str] = None, axis_index_groups: Any = None,
                ra_mean: Array = None, ra_var: Array = None ):
    """BatchNorm Module"""

    feature_axes = _canonicalize_axes(x.ndim, axis)
    reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
    feature_shape = [x.shape[ax] for ax in feature_axes]

    if (ra_mean is not None) and (ra_var is not None):
      mean, var = ra_mean.value, ra_var.value
    else:
      mean, var = _compute_stats( x, reduction_axes, dtype=dtype, axis_name=axis_name, axis_index_groups=axis_index_groups)

      if (ra_mean is not None) and (ra_var is not None) :
        ra_mean = momentum * ra_mean + (1 - momentum) * mean
        ra_var = momentum * ra_var + (1 - momentum) * var

    return _normalize( x, mean, var, reduction_axes, feature_axes,
                        dtype, param_dtype, epsilon, bias, scale)