#!/user/bw2762/.conda/envs/testbed_2/bin/python

# import numpy as np
import jax
import jax.numpy as jnp

from enn import networks
from enn import datasets
from enn import supervised
from enn import utils

# import tensorflow as tf
# from tensorflow.python.ops.nn_impl import _compute_sampled_logits
# import numpy as np
# import scipy


EPSILON = jnp.finfo(jnp.float32).tiny
 

def top_k(x,a,k):
    _ , a_khot = jax.lax.top_k(a,k)
    # print(x.y)

    subset_x = x.x[a_khot]
    subset_y = x.y[a_khot]
    # subset_x = jnp.split(x.x,a_khot)
    # subset_y = jnp.split(x.y,a_khot)

    # print(subset_y)
    # subset = x[a_khot]

    subset = datasets.ArrayBatch(x=subset_x, y=subset_y)

    return subset


def gumbel_keys(w,key):
    # sample some gumbels
    uniform = jax.random.uniform(
        key,
        w.shape,
        minval=EPSILON,
        maxval=1.0)
    z = -jnp.log(-jnp.log(uniform))
    w = w + z
    return w


def continuous_topk(w, k, t, separate=False):
    khot_list = []
    onehot_approx = jnp.zeros_like(w, dtype=jnp.float32)
    for i in range(k):
        khot_mask = jnp.maximum(1.0 - onehot_approx, EPSILON)
        w += jnp.log(khot_mask)
        onehot_approx = jax.nn.softmax(w / t, axis=-1)
        khot_list.append(onehot_approx)
    if separate:
        return khot_list
    else:
        # return jnp.sum(khot_list, 0)
        return jnp.sum(jnp.stack(khot_list), 0)


def sample_subset_continuous(w, k, key, t=0.1):
    '''
    Args:
        w (Tensor): Float Tensor of weights for each element. In gumbel mode
            these are interpreted as log probabilities
        k (int): number of elements in the subset sample
        t (float): temperature of the softmax
    '''
    w = gumbel_keys(w , key)
    return continuous_topk(w, k, t)

def sample_subset(x,w,k,key,t=0.1):
    a = sample_subset_continuous(w,k,key,t)
    _ , a_khot = jax.lax.top_k(a,k)
    subset = x[a_khot]
    return subset

# def sample_subset_from_a(x,a,k):
#     _ , a_khot = jax.lax.top_k(a,k)
#     subset = x[a_khot]
#     return subset


def weighted_reservoir_sampling_get_key(w,key):
    uniform = jax.random.uniform(
    key,
    w.shape,
    minval=EPSILON,
    maxval=1.0)
    r = uniform**(1/w)
    return r



############################################3
# From https://github.com/ermongroup/neuralsort/
##########################################

# def bl_matmul(A, B):
#     return tf.einsum('mij,jk->mik', A, B)

# def br_matmul(A, B):
#     return tf.einsum('ij,mjk->mik', A, B)

# def batchwise_matmul(A, B):
#     return tf.einsum('mij,mj->mi', A, B)

# # s: M x n x 1
# # sortnet(s): M x n x n
# def sortnet(s, tau = 1):
#   A_s = s - tf.transpose(s, perm=[0, 2, 1])
#   A_s = tf.abs(A_s)
#   # As_ij = |s_i - s_j|

#   n = tf.shape(s)[1]
#   one = tf.ones((n, 1), dtype = tf.float32)

#   B = bl_matmul(A_s, one @ tf.transpose(one))
#   # B_:k = (A_s)(one)

#   K = tf.range(n) + 1
#   # K_k = k

#   C = bl_matmul(
#     s, tf.expand_dims(tf.cast(n + 1 - 2 * K, dtype = tf.float32), 0)
#   )
#   # C_:k = (n + 1 - 2k)s

#   P = tf.transpose(C - B, perm=[0, 2, 1])
#   # P_k: = (n + 1 - 2k)s - (A_s)(one)

#   P = tf.nn.softmax(P / tau, -1)
#   # P_k: = softmax( ((n + 1 - 2k)s - (A_s)(one)) / tau )

#   return P

###############################################