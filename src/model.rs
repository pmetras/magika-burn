//! Magika `standard_v3_3`'s forward pass, written out for f32.
//!
//! One sample goes through:
//!
//! 1. an embedding of its 2048 tokens (the ONNX graph's one-hot × matrix is a row lookup) and GELU;
//! 2. a layer norm of each of the 256 channels over the 512 positions (the embeddings of four consecutive tokens
//!    form one position);
//! 3. a convolution from 256 to 512 channels, width 5, then GELU and the maximum over positions;
//! 4. a layer norm, a dense layer and a softmax.
//!
//! The convolution is almost all of the time. On x86-64 the whole pass runs through a function compiled for AVX2
//! and FMA when the processor has them, chosen once at run time, which is about two and a half times faster than
//! the baseline instruction set the crate is otherwise compiled for.
//!
//! `scripts/prepare_model.py` writes the weights and checks a NumPy version of this forward pass against ONNX
//! Runtime.

use std::sync::OnceLock;

use crate::config::{INPUT_SIZE, NUM_LABELS, PADDING_TOKEN};

/// Features per token.
const DIM: usize = 64;
/// Byte values and the padding token.
const VOCABULARY: usize = PADDING_TOKEN as usize + 1;
/// Channels of a position: the features of four consecutive tokens.
const CHANNELS: usize = 256;
const POSITIONS: usize = INPUT_SIZE * DIM / CHANNELS;
/// Output channels of the convolution.
const FILTERS: usize = 512;
const WIDTH: usize = 5;
/// Positions the convolution has a full window for.
const VALID: usize = POSITIONS - WIDTH + 1;

/// The convolution computes blocks of 4 filters × 16 positions.
const BLOCK_FILTERS: usize = 4;
const BLOCK_POSITIONS: usize = 16;
/// One channel's row of the convolution input, padded with zeros for the last block's window.
const ROW: usize = POSITIONS + BLOCK_POSITIONS;
const _: () = assert!(POSITIONS.is_multiple_of(BLOCK_POSITIONS));
const _: () = assert!(FILTERS.is_multiple_of(BLOCK_FILTERS));
const _: () = assert!(POSITIONS + WIDTH <= ROW);

/// Number of f32 values in the weights file.
const PARAMETERS: usize = VOCABULARY * DIM // emb
    + DIM // b0
    + 2 * POSITIONS // ln0_s, ln0_b
    + FILTERS * CHANNELS * WIDTH // conv_w
    + 3 * FILTERS // conv_b, ln1_s, ln1_b
    + FILTERS * NUM_LABELS // w1
    + NUM_LABELS; // b1

#[cfg(target_endian = "big")]
compile_error!("the weights are stored as little-endian f32");

#[repr(C, align(4))]
struct Aligned<T>(T);

/// The weights, aligned for reading them as f32. A file of another size does not compile.
static WEIGHT_BYTES: Aligned<[u8; PARAMETERS * 4]> = Aligned(*include_bytes!(
    "../assets/models/standard_v3_3/weights.f32"
));

/// The weight tensors, in the order of the weights file.
struct Weights {
    /// Embedding: `[VOCABULARY][DIM]`.
    emb: &'static [f32],
    b0: &'static [f32],
    /// Layer norm 0 scale and bias, per position.
    ln0_s: &'static [f32],
    ln0_b: &'static [f32],
    /// Convolution: `[FILTERS][CHANNELS][WIDTH]`.
    conv_w: &'static [f32],
    conv_b: &'static [f32],
    /// Layer norm 1 scale and bias, per filter.
    ln1_s: &'static [f32],
    ln1_b: &'static [f32],
    /// Dense layer: `[FILTERS][NUM_LABELS]`.
    w1: &'static [f32],
    b1: &'static [f32],
}

impl Weights {
    fn get() -> Self {
        // SAFETY: the bytes live for the whole program, are 4-byte aligned, and hold PARAMETERS f32 values; every
        // bit pattern is a valid f32, and the file is little-endian like the target.
        let mut rest: &'static [f32] =
            unsafe { core::slice::from_raw_parts(WEIGHT_BYTES.0.as_ptr().cast(), PARAMETERS) };
        let mut take = |len: usize| {
            let (head, tail) = rest.split_at(len);
            rest = tail;
            head
        };
        Self {
            emb: take(VOCABULARY * DIM),
            b0: take(DIM),
            ln0_s: take(POSITIONS),
            ln0_b: take(POSITIONS),
            conv_w: take(FILTERS * CHANNELS * WIDTH),
            conv_b: take(FILTERS),
            ln1_s: take(FILTERS),
            ln1_b: take(FILTERS),
            w1: take(FILTERS * NUM_LABELS),
            b1: take(NUM_LABELS),
        }
    }
}

/// Scores one sample. `tokens` are byte values or [`PADDING_TOKEN`].
pub(crate) fn forward(tokens: &[i32; INPUT_SIZE], scores: &mut [f32; NUM_LABELS]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: guarded by the runtime detection above.
        return unsafe { forward_avx2(tokens, scores) };
    }
    forward_impl::<false>(tokens, scores)
}

/// The forward pass with AVX2 and FMA. Everything it calls must inline into it:
/// `#[target_feature]` does not reach a function that stays out of line, which
/// would both lose the wider registers and turn `FUSED`'s `mul_add` into a call
/// to libm's `fmaf`, two orders of magnitude slower than the fused instruction.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn forward_avx2(tokens: &[i32; INPUT_SIZE], scores: &mut [f32; NUM_LABELS]) {
    forward_impl::<true>(tokens, scores)
}

/// `FUSED` says that `f32::mul_add` compiles to a single instruction, which is
/// only true where FMA is available.
#[inline(always)]
fn forward_impl<const FUSED: bool>(tokens: &[i32; INPUT_SIZE], scores: &mut [f32; NUM_LABELS]) {
    let w = Weights::get();
    // The embeddings after GELU, position-major.
    let mut h = zeroed::<{ POSITIONS * CHANNELS }>();
    embed(&w, tokens, &mut h);
    // The normalized convolution input, channel-major, each row padded with zeros.
    let mut x = zeroed::<{ CHANNELS * ROW }>();
    norm0(&w, &h, &mut x);
    let mut pooled = [0.0; FILTERS];
    conv::<FUSED>(&w, &x, &mut pooled);
    head(&w, &mut pooled, scores);
}

/// A zeroed buffer on the heap, whose length the compiler knows.
fn zeroed<const N: usize>() -> Box<[f32; N]> {
    vec![0.0; N]
        .into_boxed_slice()
        .try_into()
        .expect("the length is N")
}

/// GELU, tanh approximation, in the ONNX graph's order of operations.
#[inline(always)]
fn gelu(x: f32) -> f32 {
    x * (0.5 * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh()))
}

/// Embedding, bias and GELU. Token `t`'s features are channels `(t % 4) * DIM..` of position `t / 4`.
fn embed(w: &Weights, tokens: &[i32; INPUT_SIZE], h: &mut [f32; POSITIONS * CHANNELS]) {
    let table = embedding_table(w);
    let (outs, _) = h.as_chunks_mut::<DIM>();
    for (out, &token) in outs.iter_mut().zip(tokens) {
        let token = token.clamp(0, PADDING_TOKEN) as usize;
        out.copy_from_slice(&table[token * DIM..][..DIM]);
    }
}

/// `gelu(emb + b0)` for every token and feature.
///
/// Both operands are weights, so the result depends only on the token and the
/// feature, never on the content: there are `VOCABULARY * DIM` values behind the
/// [`INPUT_SIZE`] `* DIM` the embedding would otherwise compute per sample. This is
/// the same arithmetic, not an approximation, so the scores are unchanged.
fn embedding_table(w: &Weights) -> &'static [f32; VOCABULARY * DIM] {
    static TABLE: OnceLock<Box<[f32; VOCABULARY * DIM]>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = zeroed::<{ VOCABULARY * DIM }>();
        let (rows, _) = table.as_chunks_mut::<DIM>();
        for (token, row) in rows.iter_mut().enumerate() {
            let emb = &w.emb[token * DIM..][..DIM];
            for ((out, &e), &b) in row.iter_mut().zip(emb).zip(w.b0) {
                *out = gelu(e + b);
            }
        }
        table
    })
}

/// Layer norm 0: each channel over the positions, scaled and shifted per position. Writes channel-major rows.
fn norm0(w: &Weights, h: &[f32; POSITIONS * CHANNELS], x: &mut [f32; CHANNELS * ROW]) {
    let mut sum = [0.0f32; CHANNELS];
    let mut squares = [0.0f32; CHANNELS];
    let (positions, _) = h.as_chunks::<CHANNELS>();
    for position in positions {
        for ((sum, squares), &v) in sum.iter_mut().zip(&mut squares).zip(position) {
            *sum += v;
            *squares += v * v;
        }
    }
    let (rows, _) = x.as_chunks_mut::<ROW>();
    for (c, (row, (&sum, &squares))) in rows.iter_mut().zip(sum.iter().zip(&squares)).enumerate() {
        let mean = sum / POSITIONS as f32;
        let var = (squares / POSITIONS as f32 - mean * mean).max(0.0);
        let rsqrt = 1.0 / (var + 1e-6).sqrt();
        for (l, out) in row[..POSITIONS].iter_mut().enumerate() {
            *out = (h[l * CHANNELS + c] - mean) * (rsqrt * w.ln0_s[l]) + w.ln0_b[l];
        }
    }
}

/// The convolution with its bias, GELU and the maximum over the valid positions, block by block, so that the
/// `[FILTERS][VALID]` output never exists.
///
/// GELU is quasi-convex: it decreases to its minimum near -0.75 and increases after it, so over an interval it
/// is largest at one of the ends, and `max_j gelu(a_j + b)` is `max(gelu(min_j a_j + b), gelu(max_j a_j + b))`.
/// Float addition is monotonic, so the bias can be added after the reduction as well. Tracking the two extremes
/// leaves an inner loop of `min` and `max` that vectorizes, moves the `VALID` test out of it, and calls GELU
/// twice per filter instead of once per position. Taking only the maximum would be wrong: GELU is not monotonic,
/// and a filter whose outputs are all negative would pick the wrong end.
fn conv<const FUSED: bool>(w: &Weights, x: &[f32; CHANNELS * ROW], pooled: &mut [f32; FILTERS]) {
    let (pooled, _) = pooled.as_chunks_mut::<BLOCK_FILTERS>();
    for (o, best) in (0..FILTERS).step_by(BLOCK_FILTERS).zip(pooled) {
        let mut highest = [f32::NEG_INFINITY; BLOCK_FILTERS];
        let mut lowest = [f32::INFINITY; BLOCK_FILTERS];
        for p in (0..POSITIONS).step_by(BLOCK_POSITIONS) {
            let acc = block::<FUSED>(w, x, o, p);
            // Only the last block has positions without a full window.
            let valid = VALID.saturating_sub(p).min(BLOCK_POSITIONS);
            for ((highest, lowest), acc) in highest.iter_mut().zip(&mut lowest).zip(&acc) {
                let (mut hi, mut lo) = (*highest, *lowest);
                for &a in &acc[..valid] {
                    hi = hi.max(a);
                    lo = lo.min(a);
                }
                (*highest, *lowest) = (hi, lo);
            }
        }
        let biases = &w.conv_b[o..][..BLOCK_FILTERS];
        for ((best, (&hi, &lo)), &bias) in
            best.iter_mut().zip(highest.iter().zip(&lowest)).zip(biases)
        {
            *best = gelu(hi + bias).max(gelu(lo + bias));
        }
    }
}

/// Filters `o..o + BLOCK_FILTERS` at positions `p..p + BLOCK_POSITIONS`, before the bias. The fixed-length slices let
/// the compiler drop the bounds checks and vectorize the inner loop.
#[inline(always)]
fn block<const FUSED: bool>(
    w: &Weights,
    x: &[f32; CHANNELS * ROW],
    o: usize,
    p: usize,
) -> [[f32; BLOCK_POSITIONS]; BLOCK_FILTERS] {
    let mut acc = [[0.0f32; BLOCK_POSITIONS]; BLOCK_FILTERS];
    for c in 0..CHANNELS {
        let row = &x[c * ROW + p..][..BLOCK_POSITIONS + WIDTH];
        for k in 0..WIDTH {
            let xs: &[f32; BLOCK_POSITIONS] = row[k..][..BLOCK_POSITIONS].try_into().unwrap();
            for (i, acc) in acc.iter_mut().enumerate() {
                let weight = w.conv_w[((o + i) * CHANNELS + c) * WIDTH + k];
                for (acc, &x) in acc.iter_mut().zip(xs) {
                    // Only fused where FMA exists; elsewhere `mul_add` is a libm call.
                    *acc = if FUSED {
                        weight.mul_add(x, *acc)
                    } else {
                        *acc + weight * x
                    };
                }
            }
        }
    }
    acc
}

/// Layer norm 1, the dense layer and the softmax.
fn head(w: &Weights, pooled: &mut [f32; FILTERS], scores: &mut [f32; NUM_LABELS]) {
    let mean = pooled.iter().sum::<f32>() / FILTERS as f32;
    let var = (pooled.iter().map(|v| v * v).sum::<f32>() / FILTERS as f32 - mean * mean).max(0.0);
    let rsqrt = 1.0 / (var + 1e-6).sqrt();
    for ((v, &scale), &bias) in pooled.iter_mut().zip(w.ln1_s).zip(w.ln1_b) {
        *v = (*v - mean) * (rsqrt * scale) + bias;
    }

    scores.copy_from_slice(w.b1);
    let (rows, _) = w.w1.as_chunks::<NUM_LABELS>();
    for (&v, row) in pooled.iter().zip(rows) {
        for (score, &weight) in scores.iter_mut().zip(row) {
            *score += v * weight;
        }
    }

    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0;
    for score in scores.iter_mut() {
        *score = (*score - max).exp();
        total += *score;
    }
    for score in scores.iter_mut() {
        *score /= total;
    }
}
