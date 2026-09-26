"""Standalone NumPy integer RVQ contract; no training dependencies.

Query is signed A16, codewords W8, scores signed INT32. Subtraction is
17-bit (no premature clipping); stage requant uses INT64 product and
half-away rounding, then A16 saturation. Lowest index wins all ties.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np


def round_away(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def ratio_parameters(ratio):
    if not math.isfinite(ratio) or ratio <= 0 or ratio > 2**31 - 1:
        raise ValueError(f"Unrepresentable positive requant ratio: {ratio}")
    shift = min(63, max(0, math.floor(math.log2((2**31 - 1) / ratio))))
    multiplier = min(2**31 - 1, max(1, int(round_away(ratio * 2**shift))))
    return multiplier, shift


def requant(x, multiplier, shift):
    x = np.asarray(x, dtype=np.int64)
    # Subtractions and sums in this contract are bounded; detect misuse.
    if np.max(np.abs(x), initial=0) > (2**63 - 1) // multiplier:
        raise OverflowError("INT64 requant product overflow")
    product = x * multiplier
    magnitude = np.abs(product)
    # Quotient/remainder rounding avoids overflow from adding 2**62.
    quotient = magnitude >> shift
    if shift:
        quotient += (magnitude & (2**shift - 1)) >= 2**(shift - 1)
    return np.where(product < 0, -quotient, quotient)


def clip16(x):
    return np.clip(x, -32768, 32767).astype(np.int16)


def quantize_books(books, scales):
    if len(books) != len(scales) or not books:
        raise ValueError("One positive scale is required per stage")
    dim = books[0].shape[1]
    if dim > 128:
        raise ValueError("INT32 score bound only supported for dimension <=128")
    result = []
    for book, scale in zip(books, scales):
        if book.ndim != 2 or book.shape[1] != dim or not np.isfinite(book).all():
            raise ValueError("Codebooks must be finite [K,D] with equal D")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Scale must be finite and positive")
        result.append(np.clip(round_away(book / scale), -128, 127).astype(np.int8))
    return result


def lookup(indices, books, scales, output_scale):
    """Decoder contract: indices alone reproduce the exact Encoder output."""
    accumulator = np.zeros((*indices.shape[:-1], books[0].shape[1]), dtype=np.int64)
    for q, (book, scale) in enumerate(zip(books, scales)):
        selected = book[indices[..., q]].astype(np.int64)
        accumulator += requant(selected, *ratio_parameters(scale / output_scale))
    if np.max(np.abs(accumulator), initial=0) > 2**31 - 1:
        raise OverflowError("Decoder INT32 sum overflow")
    return clip16(accumulator)


def run_integer(query, books, scales, output_scale, *, squared_distance=False, residual_scales=None):
    """query already quantized to first stage scale; shape [...,D]."""
    if np.asarray(query).dtype != np.int16:
        raise ValueError("query must be INT16 at stage-0 scale")
    residual = query.astype(np.int64)
    residual_scales = scales if residual_scales is None else residual_scales
    if len(residual_scales) != len(scales):
        raise ValueError('One residual scale per stage is required')
    indices, trace = [], []
    for q, book in enumerate(books):
        stored_residual = residual.copy()
        search_raw = requant(residual, *ratio_parameters(residual_scales[q]/scales[q]))
        search_saturation = int(np.count_nonzero((search_raw < -32768)|(search_raw > 32767)))
        residual = clip16(search_raw).astype(np.int64)
        e = book.astype(np.int64)
        norm = (e * e).sum(-1)
        if squared_distance:
            # Independent exhaustive distance oracle (INT64 distance).
            distances = ((residual[..., None, :] - e)**2).sum(-1)
            index = distances.argmin(-1)
            scores = distances - (residual * residual).sum(-1)[..., None]
        else:
            scores = norm - 2 * (residual @ e.T)
            index = scores.argmin(-1)
        if scores.min() < -(2**31) or scores.max() > 2**31 - 1:
            raise OverflowError("RVQ score exceeds signed INT32")
        after = stored_residual - requant(e[index], *ratio_parameters(scales[q]/residual_scales[q]))
        if np.any((after < -65536)|(after > 65535)):
            raise OverflowError('RVQ residual subtraction exceeds signed 17-bit contract')
        following = after
        saturation = 0
        if q + 1 < len(books):
            following = requant(after, *ratio_parameters(residual_scales[q] / residual_scales[q + 1]))
            saturation = int(np.count_nonzero((following < -32768) | (following > 32767)))
            following = clip16(following).astype(np.int64)
        trace.append(dict(residual=residual.copy(), stored_residual=stored_residual,
                          search_saturation=search_saturation, scores=scores.astype(np.int32),
                          index=index, after_subtract=after, saturation=saturation))
        indices.append(index)
        residual = following
    indices = np.stack(indices, axis=-1)
    return lookup(indices, books, scales, output_scale), indices, trace


def export_package(directory, books, scales, output_scale, *, provenance, residual_scales=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    separate_residual = residual_scales is not None
    residual_scales = scales if residual_scales is None else residual_scales
    raw, packed, norms, params = bytearray(), bytearray(), bytearray(), bytearray()
    align_params = bytearray()
    layers = []
    for q, (book, scale) in enumerate(zip(books, scales)):
        k, d = book.shape
        layout = np.zeros(((k+7)//8, (d+3)//4, 4, 8), dtype=np.int8)
        for c in range(k):
            for j in range(d):
                layout[c//8, j//4, j%4, c%8] = book[c, j]
        next_params = ratio_parameters(residual_scales[q] / residual_scales[q+1]) if q+1 < len(books) else (1, 0)
        out_params = ratio_parameters(scale / output_scale)
        layers.append(dict(stage=q, codebook_size=k, codebook_dim=d, scale=float(scale),
                           codebook_offset=len(raw), packed_offset=len(packed),
                           norm_offset=len(norms), requant_offset=len(params),
                           next_multiplier=next_params[0], next_shift=next_params[1],
                           output_multiplier=out_params[0], output_shift=out_params[1]))
        raw.extend(book.tobytes())
        packed.extend(layout.tobytes())
        norms.extend((book.astype(np.int64)**2).sum(-1).astype('<i4').tobytes())
        params.extend(struct.pack('<iBiB', *next_params, *out_params))
        layers[-1]['residual_scale'] = float(residual_scales[q])
        layers[-1]['search_multiplier'], layers[-1]['search_shift'] = ratio_parameters(residual_scales[q]/scale)
        layers[-1]['subtract_multiplier'], layers[-1]['subtract_shift'] = ratio_parameters(scale/residual_scales[q])
        align_params.extend(struct.pack('<iBiB', *ratio_parameters(residual_scales[q]/scale), *ratio_parameters(scale/residual_scales[q])))
    blobs = dict(rvq_codebook_int8=raw, rvq_codebook_packed_4x8_int8=packed,
                 rvq_codebook_norm_int32=norms, rvq_stage_requant_params=params)
    if separate_residual:
        blobs['rvq_search_subtract_requant_params'] = align_params
    for name, blob in blobs.items():
        (directory / (name+'.bin')).write_bytes(blob)
    manifest = dict(format_version=2 if separate_residual else 1, num_quantizers=len(books), stages=layers,
                    codebook_bits=8, residual_bits=16, distance_accumulator_bits=32,
                    subtraction_bits=17, requant_product_bits=64, zero_point=0,
                    rounding='half_away_from_zero', tie_break='lowest_index',
                    input_scale=float(residual_scales[0]), output_scale=float(output_scale),
                    search_subtract_parameter_layout='<iBiB: residual-to-search M,S; codeword-to-residual M,S; 10 bytes/stage',
                    input_contract='INT16 lookup-space query at input_scale',
                    output_contract='sum requanted selected codewords in INT32 then saturate INT16',
                    parameter_layout='<iB iB: next_multiplier,next_shift,output_multiplier,output_shift; 10 bytes/stage',
                    packed_layout='output_group,reduction_group,4,8; padded entries zero; argmin only valid K',
                    projection_quantized=False, provenance=provenance,
                    sha256={n+'.bin': hashlib.sha256(b).hexdigest() for n,b in blobs.items()})
    (directory/'rvq_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
