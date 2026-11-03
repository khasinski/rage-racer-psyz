#ifndef PSYZ_VRAM_SAMPLE_MASK_H
#define PSYZ_VRAM_SAMPLE_MASK_H
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* Conservative 32x32 word tiles, independent of readback/cache bookkeeping. */
typedef struct { uint32_t rows[16]; } VramSampleMask;
typedef struct { uint32_t columns; int first, last; } VramSampleRegion;

static inline VramSampleRegion VramSampleRegionFor(int x, int y, int w, int h) {
    VramSampleRegion empty = {0, 0, -1};
    if (w <= 0 || h <= 0) return empty;
    int64_t right = (int64_t)x + w, bottom = (int64_t)y + h;
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (right > 1024) right = 1024;
    if (bottom > 512) bottom = 512;
    if (x >= right || y >= bottom) return empty;
    unsigned first = (unsigned)x / 32, last = (unsigned)(right - 1) / 32;
    return (VramSampleRegion){(UINT32_MAX << first) & (UINT32_MAX >> (31 - last)),
        y / 32, (int)(bottom - 1) / 32};
}

static inline void VramSampleMaskReset(VramSampleMask *mask, bool dirty) {
    memset(mask, dirty ? 255 : 0, sizeof(*mask));
}
static inline void VramSampleMaskMark(VramSampleMask *mask, int x, int y, int w, int h) {
    VramSampleRegion r = VramSampleRegionFor(x, y, w, h);
    for (int row = r.first; row <= r.last; ++row) mask->rows[row] |= r.columns;
}
static inline bool VramSampleMaskIntersects(const VramSampleMask *mask,
                                           int x, int y, int w, int h) {
    VramSampleRegion r = VramSampleRegionFor(x, y, w, h);
    for (int row = r.first; row <= r.last; ++row)
        if (mask->rows[row] & r.columns) return true;
    return false;
}

/* Match the shader's full possible page and CLUT footprint, including texture
 * windows. Conservative pages avoid dependence on interpolated UV bounds. */
static inline bool VramSampleMaskTextureDirty(const VramSampleMask *mask,
                                              uint16_t tpage, uint16_t clut) {
    if (tpage & 0x8000) return false;
    unsigned depth = (tpage >> 7) & 3;
    int width = depth == 0 ? 64 : depth == 1 ? 128 : 256;
    if (VramSampleMaskIntersects(mask, (tpage & 15) * 64,
                                (tpage & 16) ? 256 : 0, width, 256)) return true;
    return depth < 2 && VramSampleMaskIntersects(mask, (clut & 63) * 16,
                                                clut >> 6, depth ? 256 : 16, 1);
}
#endif
