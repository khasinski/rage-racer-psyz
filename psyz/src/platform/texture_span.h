#ifndef PSYZ_TEXTURE_SPAN_H
#define PSYZ_TEXTURE_SPAN_H
#include <stdint.h>

typedef struct {
    int x_start, x_end;
    double x_left, x_right;
    double u_left, u_right;
    double v_left, v_right;
    double r_left, r_right;
    double g_left, g_right;
    double b_left, b_right;
} RasterTextureSpan;

typedef struct {
    int x_start;
    int fixed_u, fixed_v, fixed_r, fixed_g, fixed_b;
    int step_u, step_v, step_r, step_g, step_b;
} PreparedTextureSpan;

/* Preserve truncation and the original 16.16 arithmetic exactly. The
 * derivatives depend on the scanline, not on the sampled pixel. */
static inline PreparedTextureSpan TextureSpanPrepare(const RasterTextureSpan *span) {
    double width = span->x_right - span->x_left;
    double start_t = width > 0.0 ?
        (span->x_start - span->x_left) / width : 0.0;
    double start_u = span->u_left +
        (span->u_right - span->u_left) * start_t;
    double start_v = span->v_left +
        (span->v_right - span->v_left) * start_t;
    int fixed_u = (int)(start_u * 65536.0);
    int fixed_v = (int)(start_v * 65536.0);
    int step_u = width > 0.0 ?
        (int)((span->u_right - span->u_left) / width * 65536.0) : 0;
    int step_v = width > 0.0 ?
        (int)((span->v_right - span->v_left) / width * 65536.0) : 0;
    int fixed_r = (int)((span->r_left +
        (span->r_right - span->r_left) * start_t) * 65536.0);
    int fixed_g = (int)((span->g_left +
        (span->g_right - span->g_left) * start_t) * 65536.0);
    int fixed_b = (int)((span->b_left +
        (span->b_right - span->b_left) * start_t) * 65536.0);
    int step_r = width > 0.0 ?
        (int)((span->r_right - span->r_left) / width * 65536.0) : 0;
    int step_g = width > 0.0 ?
        (int)((span->g_right - span->g_left) / width * 65536.0) : 0;
    int step_b = width > 0.0 ?
        (int)((span->b_right - span->b_left) / width * 65536.0) : 0;
    return (PreparedTextureSpan){span->x_start,
        fixed_u, fixed_v, fixed_r, fixed_g, fixed_b,
        step_u, step_v, step_r, step_g, step_b};
}

/* Coverage/texel comparison needs only UV. Defer colour interpolation until
 * a compatibility pixel is actually required. */
static inline void TextureSpanSampleUV(const PreparedTextureSpan *span, int x,
        uint16_t *u, uint16_t *v) {
    int fixed_u = span->fixed_u, step_u = span->step_u;
    int fixed_v = span->fixed_v, step_v = span->step_v;
    fixed_u += (x - span->x_start) * step_u;
    fixed_v += (x - span->x_start) * step_v;
    *u = (uint16_t)((fixed_u >> 16) & 0xff);
    *v = (uint16_t)((fixed_v >> 16) & 0xff);
}

static inline void TextureSpanSampleColor(const PreparedTextureSpan *span, int x,
        uint8_t *r, uint8_t *g, uint8_t *b) {
    int fixed_r = span->fixed_r, step_r = span->step_r;
    int fixed_g = span->fixed_g, step_g = span->step_g;
    int fixed_b = span->fixed_b, step_b = span->step_b;
    fixed_r += (x - span->x_start) * step_r;
    fixed_g += (x - span->x_start) * step_g;
    fixed_b += (x - span->x_start) * step_b;
    *r = (uint8_t)((fixed_r >> 16) < 0 ? 0 : (fixed_r >> 16) > 255 ? 255 : (fixed_r >> 16));
    *g = (uint8_t)((fixed_g >> 16) < 0 ? 0 : (fixed_g >> 16) > 255 ? 255 : (fixed_g >> 16));
    *b = (uint8_t)((fixed_b >> 16) < 0 ? 0 : (fixed_b >> 16) > 255 ? 255 : (fixed_b >> 16));
}

static inline void TextureSpanSample(const PreparedTextureSpan *span, int x,
        uint16_t *u, uint16_t *v, uint8_t *r, uint8_t *g, uint8_t *b) {
    TextureSpanSampleUV(span, x, u, v);
    TextureSpanSampleColor(span, x, r, g, b);
}

/* A forward unit-UV row with constant V/color can use one rectangle. Stop
 * before byte UV wrap; the caller also bounds it by triangle ownership. */
static inline int TextureSpanUnitRun(const PreparedTextureSpan *span,
                                    uint16_t first_u, int remaining) {
    if (remaining < 1) return 0;
    if (span->step_u != 65536 || span->step_v || span->step_r ||
        span->step_g || span->step_b || first_u > 255) return 1;
    int before_wrap = 256 - first_u;
    return remaining < before_wrap ? remaining : before_wrap;
}

/* A fractional fixed-point step can still select consecutive texels. Bound
 * the run by the first carry/borrow relative to a unit step, preserving the
 * exact integer sampler rather than approximating its interpolation. */
static inline int TextureSpanConsecutiveRun(const PreparedTextureSpan *span,
        int x, uint16_t first_u, int remaining) {
    if (span->step_u == 65536)
        return TextureSpanUnitRun(span, first_u, remaining);
    if (remaining < 1) return 0;
    if (span->step_v || span->step_r || span->step_g || span->step_b ||
        first_u > 255) return 1;
    int count = remaining < 256 - first_u ? remaining : 256 - first_u;
    int64_t fixed = (int64_t)span->fixed_u +
        ((int64_t)x - span->x_start) * span->step_u;
    int64_t fraction = (uint64_t)fixed & 65535;
    int64_t drift = (int64_t)span->step_u - 65536;
    int64_t limit = drift > 0 ? (65535 - fraction) / drift + 1 :
        drift < 0 ? fraction / -drift + 1 : count;
    return limit < count ? (int)limit : count;
}
#endif
