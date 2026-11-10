// SPDX-FileCopyrightText: 2026 the XeMaya contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/minimax.hpp - MiniMax-M3's (minimax-m3) own pieces: MiniMax Sparse Attention (MSA).
//
// After llama.cpp's src/models/minimax-m3.cpp (master 8642541b2, MIT): a block that routes experts scores every
// visible position with its indexer (one indexer head a K/V head), a block of `blk` positions takes its best
// position's score, and each K/V head's queries attend the cells of its `top_k` best blocks, the `local` blocks ending
// at the query's own always among them.  A query that sees at most top_k blocks attends every cell up to its own: the
// attention of the leading dense blocks, and of every block while the context is top_k * blk tokens or less.
//
// The K/V cache is the plain models' (qsa.hpp: FP16 pools `[page][kv_head][page_size][head_dim]` through a page
// table, a cell a position); the indexer's keys are one FP16 row a cell.  A token's position is its step record's
// kStepPos (device memory, so one recorded graph serves every position).
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

/// The indexer's keys of n_tok tokens (normed and rotated; ik [n_tok][dim] FP32) into `cache` [cell][dim] FP16, token
/// t's at the cell step[t * step_stride + kStepPos].
void mm_idx_append(uint16_t* cache, const int32_t* step, int step_stride, const float* ik, int dim, int n_tok,
                   void* stream);

/// The blocks each K/V head of each token attends.  iq [n_tok][n_kv][dim] FP32 (normed and rotated).  A token at
/// position p sees blocks 0 .. p / blk; when those are at most top_k it takes them all, else the `local` blocks ending
/// at its own and the best of the others by the largest iq . key over a block's visible cells (ties: the lower block).
/// sel [n_tok][n_kv][top_k] the blocks, ascending; count [n_tok][n_kv] how many.  `scores` is scratch,
/// [n_tok][n_kv][max_blocks] floats; max_blocks at least the blocks any of the tokens sees (it sizes the grid).
void mm_msa_select(const float* iq, const uint16_t* cache, const int32_t* step, int step_stride, int n_tok, int n_kv,
                   int dim, int blk, int top_k, int local, int64_t max_blocks, float* scores, int32_t* sel,
                   int32_t* count, void* stream);

/// Scratch floats a token for mm_msa_attn with `splits` (1: none).
uint64_t mm_msa_attn_scratch_floats(int splits, const QsaShapes& s);

/// attn[t][h] = softmax(q[t][h] . K[c][h / G] / sqrt(head_dim)) . V over the cells c <= p of the blocks sel lists for
/// (t, h / G) (`sel` null: every cell c <= p), q and attn [n_tok][n_head][head_dim] FP32, K and V the FP16 pools.
/// `splits` > 1 (decode windows): a query's cells in that many parts over work-items, merged from `scratch`
/// (n_tok * mm_msa_attn_scratch_floats floats); 1 (prompt chunks): a work-item a query, no scratch.
void mm_msa_attn(const float* q, const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table,
                 const int32_t* step, int step_stride, const int32_t* sel, const int32_t* count, int top_k, int blk,
                 const QsaShapes& s, int splits, float* scratch, float* attn, int64_t n_tok, void* stream);

/// mm_msa_attn on Xe's matrix engines (FP16 Q . K^T and P . V, q and p as hi + lo; glm5_mla_xmx.cpp's scheme): a
/// work-group takes a token's K/V head and the 16 query heads that share it.  False, nothing launched, where the
/// device has no FP16 matrix engine or too little local memory, or for another shape (head_dim 128, 16 query heads a
/// K/V head): the caller keeps mm_msa_attn.
bool mm_msa_attn_xmx(const float* q, const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table,
                     const int32_t* step, int step_stride, const int32_t* sel, const int32_t* count, int top_k,
                     int blk, const QsaShapes& s, float* attn, int64_t n_tok, void* stream);
/// XEMAYA_MM_ATTN=ref keeps mm_msa_attn (default: xmx where the device takes it).
bool mm_msa_attn_xmx_on();

}  // namespace strata::kernels
