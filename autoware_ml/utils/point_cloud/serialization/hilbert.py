"""Hilbert-order serialization helpers for point-cloud models."""

from __future__ import annotations

import torch


def right_shift(binary, k=1, axis=-1):
    """Right shift an array of binary values."""
    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    return torch.nn.functional.pad(binary[tuple(slicing)], (k, 0), mode="constant", value=0)


def binary2gray(binary, axis=-1):
    """Convert binary values into Gray codes."""
    return torch.logical_xor(binary, right_shift(binary, axis=axis))


def gray2binary(gray, axis=-1):
    """Convert Gray codes back into binary values."""
    shift = 2 ** (torch.Tensor([gray.shape[axis]]).log2().ceil().int() - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, right_shift(gray, shift))
        shift = torch.div(shift, 2, rounding_mode="floor")
    return gray


def encode(locs, num_dims, num_bits):
    """Encode locations in a hypercube into Hilbert integers."""
    orig_shape = locs.shape
    bitpack_mask = 1 << torch.arange(0, 8).to(locs.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)

    if orig_shape[-1] != num_dims:
        raise ValueError(
            f"The last locs dimension must match num_dims, got {orig_shape[-1]} and {num_dims}."
        )
    if num_dims * num_bits > 63:
        raise ValueError(
            f"num_dims={num_dims} and num_bits={num_bits} exceed the supported int64 Hilbert range."
        )

    locs_uint8 = locs.long().view(torch.uint8).reshape((-1, num_dims, 8)).flip(-1)
    gray = (
        locs_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[..., -num_bits:]
    )

    for bit in range(0, num_bits):
        for dim in range(0, num_dims):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], mask[:, None])
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(gray[:, dim, bit + 1 :], to_flip)
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    gray = gray.swapaxes(1, 2).reshape((-1, num_bits * num_dims))
    hh_bin = gray2binary(gray)
    extra_dims = 64 - num_bits * num_dims
    padded = torch.nn.functional.pad(hh_bin, (extra_dims, 0), "constant", 0)
    hh_uint8 = (
        (padded.flip(-1).reshape((-1, 8, 8)) * bitpack_mask).sum(2).squeeze().type(torch.uint8)
    )
    return hh_uint8.view(torch.int64).squeeze()


def decode(hilberts, num_dims, num_bits):
    """Decode Hilbert integers into hypercube locations."""
    if num_dims * num_bits > 64:
        raise ValueError(
            f"num_dims={num_dims} and num_bits={num_bits} exceed the supported uint64 Hilbert range."
        )

    hilberts = torch.atleast_1d(hilberts)
    orig_shape = hilberts.shape
    bitpack_mask = 2 ** torch.arange(0, 8).to(hilberts.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)
    hh_uint8 = hilberts.ravel().type(torch.int64).view(torch.uint8).reshape((-1, 8)).flip(-1)
    hh_bits = (
        hh_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[:, -num_dims * num_bits :]
    )
    gray = binary2gray(hh_bits)
    gray = gray.reshape((-1, num_bits, num_dims)).swapaxes(1, 2)

    for bit in range(num_bits - 1, -1, -1):
        for dim in range(num_dims - 1, -1, -1):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], mask[:, None])
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(gray[:, dim, bit + 1 :], to_flip)
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    extra_dims = 64 - num_bits
    padded = torch.nn.functional.pad(gray, (extra_dims, 0), "constant", 0)
    locs_chopped = padded.flip(-1).reshape((-1, num_dims, 8, 8))
    locs_uint8 = (locs_chopped * bitpack_mask).sum(3).squeeze().type(torch.uint8)
    return locs_uint8.view(torch.int64).reshape((*orig_shape, num_dims))
