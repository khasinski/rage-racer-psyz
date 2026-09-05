#ifndef PSYZ_TRIANGLE_COVERAGE_H
#define PSYZ_TRIANGLE_COVERAGE_H
#include <stdbool.h>
#include <stdint.h>

typedef struct { int x, y; } RasterPoint;
typedef struct {
    int64_t dx[3], dy[3], constant[3];
    bool inclusive[3], degenerate;
} PreparedTriangleCoverage;

static inline PreparedTriangleCoverage TriangleCoveragePrepare(const RasterPoint p[3]) {
    PreparedTriangleCoverage result = {0};
    int64_t area = (int64_t)(p[1].x - p[0].x) * (p[2].y - p[0].y) -
                   (int64_t)(p[1].y - p[0].y) * (p[2].x - p[0].x);
    result.degenerate = area == 0;
    for (int i = 0; i < 3; ++i) {
        RasterPoint a = p[i], b = p[(i + 1) % 3];
        int64_t dx = b.x - a.x, dy = b.y - a.y;
        if (area < 0) { dx = -dx; dy = -dy; }
        result.dx[i] = -dy;
        result.dy[i] = dx;
        result.constant[i] = dy * a.x - dx * a.y;
        result.inclusive[i] = dy < 0 || (dy == 0 && dx > 0);
    }
    return result;
}

static inline bool TriangleCoverageContains(
        const PreparedTriangleCoverage *p, int x, int y) {
    if (p->degenerate) return false;
    int zero_edges = 0;
    for (int i = 0; i < 3; ++i) {
        int64_t edge = p->dx[i] * x + p->dy[i] * y + p->constant[i];
        if (edge < 0 || (edge == 0 && !p->inclusive[i])) return false;
        zero_edges += edge == 0;
    }
    /* Preserve the compatibility path's rejection of exact vertices. */
    return zero_edges < 2;
}
#endif
