#ifndef PSYZ_TEXTURE_SAMPLE_H
#define PSYZ_TEXTURE_SAMPLE_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct {
    int x, y, u, v;
} TextureSampleVertex;

/* A positive, axis-aligned 1:1 sprite has exactly integral UVs at every
 * destination pixel. Both triangle planes therefore always request the
 * compatibility correction, regardless of edge coverage. Reject wrapped
 * coordinates/UVs and non-unit mappings so they keep the general path. */
static inline bool TextureSampleIsUnitSprite(const TextureSampleVertex p[4]) {
    int w = p[1].x - p[0].x, h = p[2].y - p[0].y;
    return w > 0 && h > 0 &&
        p[1].y == p[0].y && p[2].x == p[0].x &&
        p[3].x == p[1].x && p[3].y == p[2].y &&
        p[1].u - p[0].u == w && p[1].v == p[0].v &&
        p[2].u == p[0].u && p[2].v - p[0].v == h &&
        p[3].u == p[1].u && p[3].v == p[2].v;
}

typedef struct {
    double ax, ay, u, v, du_dx, du_dy, dv_dx, dv_dy;
    bool degenerate;
} TextureSamplePlane;

/* Keep the original arithmetic order: changing it near an integral UV can
 * change both the texel and whether a compatibility pixel is required. */
static inline TextureSamplePlane TextureSamplePrepare(
        const TextureSampleVertex p[3]) {
    double ax = p[0].x, ay = p[0].y;
    double bx = p[1].x, by = p[1].y;
    double cx = p[2].x, cy = p[2].y;
    double determinant = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    TextureSamplePlane result = {0};
    result.ax = ax; result.ay = ay;
    result.u = p[0].u; result.v = p[0].v;
    result.degenerate = determinant == 0.0;
    if (result.degenerate) return result;
    result.du_dx = ((p[1].u - p[0].u) * (cy - ay) -
                    (p[2].u - p[0].u) * (by - ay)) / determinant;
    result.du_dy = ((bx - ax) * (p[2].u - p[0].u) -
                    (cx - ax) * (p[1].u - p[0].u)) / determinant;
    result.dv_dx = ((p[1].v - p[0].v) * (cy - ay) -
                    (p[2].v - p[0].v) * (by - ay)) / determinant;
    result.dv_dy = ((bx - ax) * (p[2].v - p[0].v) -
                    (cx - ax) * (p[1].v - p[0].v)) / determinant;
    return result;
}

static inline bool TextureSampleAt(const TextureSamplePlane *p, int x, int y,
                                   uint16_t *u, uint16_t *v) {
    if (p->degenerate) {
        *u = (uint16_t)p->u; *v = (uint16_t)p->v;
        return true;
    }
    double raw_u = p->u + p->du_dx * (x - p->ax) + p->du_dy * (y - p->ay);
    double raw_v = p->v + p->dv_dx * (x - p->ax) + p->dv_dy * (y - p->ay);
    int sample_u = (int)floor(raw_u), sample_v = (int)floor(raw_v);
    *u = (uint16_t)(sample_u < 0 ? 0 : sample_u > 255 ? 255 : sample_u);
    *v = (uint16_t)(sample_v < 0 ? 0 : sample_v > 255 ? 255 : sample_v);
    return fabs(raw_u - round(raw_u)) < 1e-7 ||
           fabs(raw_v - round(raw_v)) < 1e-7;
}

#endif
