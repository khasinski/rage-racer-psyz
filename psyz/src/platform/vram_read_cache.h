#ifndef PSYZ_VRAM_READ_CACHE_H
#define PSYZ_VRAM_READ_CACHE_H
#include <stdint.h>
#include <string.h>

enum { VRAM_CACHE_W = 1024, VRAM_CACHE_H = 512 };
typedef struct {
    uint16_t words[VRAM_CACHE_W * VRAM_CACHE_H];
    uint8_t valid[VRAM_CACHE_W * VRAM_CACHE_H];
} VramReadCache;

static inline int VramReadCacheRect(int x, int y, int w, int h) {
    return x >= 0 && y >= 0 && w > 0 && h > 0 &&
        x < VRAM_CACHE_W && y < VRAM_CACHE_H &&
        w <= VRAM_CACHE_W - x && h <= VRAM_CACHE_H - y;
}
static inline void VramReadCacheReset(VramReadCache *c) {
    memset(c->valid, 0, sizeof(c->valid));
}
static inline void VramReadCacheInvalidate(VramReadCache *c, int x, int y, int w, int h) {
    if (!VramReadCacheRect(x,y,w,h)) { VramReadCacheReset(c); return; }
    for (int row = 0; row < h; ++row)
        memset(c->valid + (y + row) * VRAM_CACHE_W + x, 0, (size_t)w);
}
static inline void VramReadCacheWrite(VramReadCache *c, int x, int y, int w, int h,
                                     const uint16_t *words) {
    if (!VramReadCacheRect(x,y,w,h)) return;
    for (int row = 0; row < h; ++row) {
        size_t offset = (size_t)(y + row) * VRAM_CACHE_W + x;
        memcpy(c->words + offset, words + (size_t)row * w, (size_t)w * sizeof(*words));
        memset(c->valid + offset, 1, (size_t)w);
    }
}
static inline int VramReadCacheRead(const VramReadCache *c, int x, int y, int w, int h,
                                   uint16_t *words) {
    if (!VramReadCacheRect(x,y,w,h)) return 0;
    for (int row = 0; row < h; ++row)
        for (int col = 0; col < w; ++col)
            if (!c->valid[(y + row) * VRAM_CACHE_W + x + col]) return 0;
    for (int row = 0; row < h; ++row)
        memcpy(words + (size_t)row * w, c->words + (y + row) * VRAM_CACHE_W + x,
               (size_t)w * sizeof(*words));
    return 1;
}
#endif
