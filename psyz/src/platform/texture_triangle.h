#ifndef PSYZ_TEXTURE_TRIANGLE_H
#define PSYZ_TEXTURE_TRIANGLE_H
#include <math.h>
#include <stdbool.h>
#include "texture_span.h"

typedef struct { int x, y, u, v, r, g, b; } TextureTriangleVertex;
typedef struct {
    TextureTriangleVertex vertices[3];
    int offset_x, offset_y;
} PreparedTextureTriangle;

/* Stable ordering is invariant across scanlines. Preserve equal-Y order and
 * all span interpolation/division operations from the original rasterizer. */
static inline PreparedTextureTriangle TextureTrianglePrepare(
        const TextureTriangleVertex input[3], int offset_x, int offset_y) {
    PreparedTextureTriangle result = {{input[0], input[1], input[2]},
                                      offset_x, offset_y};
    for (int i = 1; i < 3; ++i) {
        TextureTriangleVertex key = result.vertices[i];
        int j = i;
        while (j > 0 && result.vertices[j - 1].y > key.y) {
            result.vertices[j] = result.vertices[j - 1];
            --j;
        }
        result.vertices[j] = key;
    }
    return result;
}

static inline bool TextureTriangleSpanAt(const PreparedTextureTriangle *p,
                                         int y, RasterTextureSpan *span) {
    const TextureTriangleVertex *v = p->vertices;
    int y0 = v[0].y + p->offset_y;
    int y1 = v[1].y + p->offset_y;
    int y2 = v[2].y + p->offset_y;
    if (y2 == y0 || y < y0 || y > y2) return false;

    double t1;
    if (y < y1) {
        if (y1 == y0) return false;
        t1 = (double)(y - y0) / (double)(y1 - y0);
        span->x_left = v[0].x + p->offset_x + (v[1].x - v[0].x) * t1;
        span->u_left = v[0].u + (v[1].u - v[0].u) * t1;
        span->v_left = v[0].v + (v[1].v - v[0].v) * t1;
        span->r_left = (int)(v[0].r + (v[1].r - v[0].r) * t1);
        span->g_left = (int)(v[0].g + (v[1].g - v[0].g) * t1);
        span->b_left = (int)(v[0].b + (v[1].b - v[0].b) * t1);
    } else {
        if (y2 == y1) return false;
        t1 = (double)(y - y1) / (double)(y2 - y1);
        span->x_left = v[1].x + p->offset_x + (v[2].x - v[1].x) * t1;
        span->u_left = v[1].u + (v[2].u - v[1].u) * t1;
        span->v_left = v[1].v + (v[2].v - v[1].v) * t1;
        span->r_left = (int)(v[1].r + (v[2].r - v[1].r) * t1);
        span->g_left = (int)(v[1].g + (v[2].g - v[1].g) * t1);
        span->b_left = (int)(v[1].b + (v[2].b - v[1].b) * t1);
    }
    double t2 = (double)(y - y0) / (double)(y2 - y0);
    span->x_right = v[0].x + p->offset_x + (v[2].x - v[0].x) * t2;
    span->u_right = v[0].u + (v[2].u - v[0].u) * t2;
    span->v_right = v[0].v + (v[2].v - v[0].v) * t2;
    span->r_right = (int)(v[0].r + (v[2].r - v[0].r) * t2);
    span->g_right = (int)(v[0].g + (v[2].g - v[0].g) * t2);
    span->b_right = (int)(v[0].b + (v[2].b - v[0].b) * t2);
    if (span->x_left > span->x_right) {
        double swap = span->x_left;
        span->x_left = span->x_right;
        span->x_right = swap;
        swap = span->u_left;
        span->u_left = span->u_right;
        span->u_right = swap;
        swap = span->v_left;
        span->v_left = span->v_right;
        span->v_right = swap;
        swap = span->r_left;
        span->r_left = span->r_right;
        span->r_right = swap;
        swap = span->g_left;
        span->g_left = span->g_right;
        span->g_right = swap;
        swap = span->b_left;
        span->b_left = span->b_right;
        span->b_right = swap;
    }
    span->x_start = (int)ceil(span->x_left);
    span->x_end = (int)floor(span->x_right);
    return span->x_start <= span->x_end;
}
#endif

