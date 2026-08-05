// PSP GPU backend using Graphics Engine.
//
// The backend has a strong emphasis on parallelizing CPU and GPU when possible.
// DMA transfers and GPU draws are preferred when moving graphics, so the CPU
// is free to continue emulating the PS1 GPU. PS1 games are not heavy, and the
// PSP itself is powerful enough to run those games natively. The challenge of
// this backend is synchronization and accuracy.
//
// The VRAM is shadowed on RAM via g_vram. It splits into 32 tpages like on
// PSX. When the backend receives a LoadImage, it updates the shadowed RAM
// and mark the correspondent tpage as dirty. The actual upload to VRAM is
// only executed when the tpage is requested by one of the coming PS1
// primitives. Meanwhile, a StoreImage will not stall the GPU and will fetch
// the data directly from the shadowed RAM. This optimization has the
// disadvantage of GPU draws on the real VRAM tpages to not be reflected
// back to the shadowed RAM. There is also an additional 1MB+ RAM used,
// but PSP has plenty of RAM anyway.
//
// When PSYZ_ASPECT_DISPLAY is used (default), it forces the output to be
// 320x240. The game gets rendered to a temporary location in the EDRAM,
// then it gets used as a texture to blit to the display at 320x240.
// Alternatively, PSYZ_ASPECT_SQUARE completely bypasses this mechanic, and
// it renders straight to the buffer meant to show to the display. This way is
// more performant than PSYZ_ASPECT_DISPLAY, and it's pixel perfect.
//
// The double-buffering of PS1 by swapping DISP/DRAW is also replicated here.
// A collection of command list is collected from Draw_PushPrim to the ring
// buffer g_vring, then dispatched when VSync is requested. The side of the
// ring buffer then flips with dlist_idx to start filling the new buffer while
// the GPU is busy to render the previous frame. This mechanic avoids the CPU
// having to synchronize with the GPU at the cost of deferring one frame. It
// has the same 16.66ms latency of the PS1, meaning virtually no latency is
// effectively introduced compared to a real PS1 hardware.
//
// Primitives are batched in their own batch_prim, which adds on top of the
// internal primitive buffer of sceGu. It's more complex and consumes memory,
// but it also saves performance cost than calling sceGuDrawArray on every
// primitive. An alternate approach would be to byapss sceGu and build sceGe
// commands directly, which should further save performance and memory.

#include <pspdisplay.h>
#include <pspge.h>
#include <pspgu.h>
#include <psputils.h>
#include <psyz.h>
#include <psyz/log.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libgpu.h>
#include "../draw.h"

#define PSX_VRAM_W 1024
#define PSX_VRAM_H 512
#define PSP_SCREEN_W 480
#define PSP_SCREEN_H 272
#define FB_STRIDE 512

#define VRAM_TILE_W 64 // 64 for 16bpp, 128 for 8bpp, 256 for 4bpp
#define VRAM_TILE_H 256
#define VRAM_TILES_X 16
#define VRAM_TILES 32
#define VRAM_TILE_BYTES (VRAM_TILE_W * VRAM_TILE_H * 2)

static u16 g_vram[VRAM_TILES][VRAM_TILE_W * VRAM_TILE_H];
static bool tile_dirty[VRAM_TILES];

// EDRAM layout (2MB): double-buffered 512x272 16-bit framebuffers, then the
// 1MB VRAM tile mirror (the texture caches live in main RAM)
#define EDRAM_DRAW_OFFSET 0
#define EDRAM_DISP_OFFSET (FB_STRIDE * PSP_SCREEN_H * 2)
#define EDRAM_TILES_OFFSET (EDRAM_DISP_OFFSET * 2)
// off-screen surface for the aspect-corrected present path, after the mirror
#define EDRAM_SURFACE_OFFSET (EDRAM_TILES_OFFSET + VRAM_TILES * VRAM_TILE_BYTES)

static bool is_init = false;

static unsigned int __attribute__((aligned(64))) dlist[2][0x80000 / 4];
static int dlist_idx;
// stop emitting when the list is nearly full instead of corrupting memory
#define DLIST_WATERMARK (sizeof(dlist[0]) - 0x8000)

// ===== video presentation options =====
static PsyzDitherMode dither_mode = PSYZ_DITHER_AUTO;
static PsyzAspectMode aspect_mode = PSYZ_ASPECT_DISPLAY;
static PsyzVsyncMode vsync_mode = PSYZ_VSYNC_AUTO;
static unsigned internal_res = 1;

// ===== frame state =====
static u16 cur_tpage;
static bool env_dither;
static bool can_dither;
static struct {
    int x, y;
} draw_offset, draw_area_start, draw_area_end, fb_origin;
static bool fb_origin_locked;
static bool display_enabled = true;
static bool warned_list_budget;
static int list_budget_calls;
static bool list_budget_cached = true;
static struct {
    int x, y;
} disp_origins[2];
static int disp_origin_count;

// ===== display defaults, same as a real PS1 =====
static int game_w = 256;         // GP1 PS1 register for display mode
static int game_h = 240;         // GP1 PS1 register for display mode
static int set_disp_horiz = 256; // Display range for aspect ratio
static int set_disp_vert = 240;  // Display range for aspect ratio
static bool is_pal = false; // I don't think we can ever run at 50fps on PSP

// Variables to draw pixel-perfect first, then scale aspect ratio horizontally
static int out_x, out_y, out_w = PSP_SCREEN_W, out_h = PSP_SCREEN_H;
static bool use_surface;   // false: draw straight to the framebuffer
static int surface_stride; // (game_w + 7) & ~7
static int surface_slot_bytes;
static int surface_idx;
static int target_w;       // presented width after horizontal aspect scale
static int pres_x, pres_y; // presented top-left; negative = centered crop
static int map_ofs_x, map_ofs_y; // base vertex pos, passthrough only
static PsyzRect draw_area_rect;  // {0,0,0,0} = centered (default)
static int cur_draw_fb;          // which framebuffer the GE draws to (0/1)
// framebuffer index of the frame finished but not yet displayed
static int show_fb = -1;
static int applied_draw_target = -1; // last EDRAM offset given to the GE list
static bool warned_surface_budget;

// ===== cached GE render state, to skip redundant commands =====
static int cur_blend = -1; // -1 none, 0-3 = PS1 ABR mode
static int cur_dither = -1;
static int cur_textured = -1;

static void FlushBatch(void);
static int EdramSurfaceOffset(int idx); // EDRAM offset of aspect surface slot
static void WaitPrevFrameGpu(void);
static void ShowPendingFrame(void); // display the deferred finished frame
static int FbOffset(int idx);
// set when a frame's GE list has been kicked but not yet waited on
static bool prev_frame_pending;
// GE queue id of that list, captured before sceGuStart overwrites it
static int prev_frame_list_id;
// set when a finished frame is waiting to be shown
static bool pending_show;
static unsigned int last_vsync; // vblank count at the last present, for pacing

// ===== GU_DIRECT packet emission to bypass sceGu wrappers =====
typedef struct {
    unsigned int* start;
    unsigned int* current;
    int parent_context;
} GuDisplayListRef;
extern GuDisplayListRef* gu_list;
extern int ge_list_executed[2]; // pspgu: the GE list id kicked by sceGuStart

// GE command opcodes (top byte of each command word)
#define GECMD_BASE 0x10
#define GECMD_VADR 0x01
#define GECMD_PRIM 0x04
#define GECMD_VTYPE 0x12
#define GECMD_JUMP 0x08
#define GECMD_CALL 0x0A

static inline void GeCmd(unsigned int cmd, unsigned int arg) {
    *(gu_list->current++) = (cmd << 24) | (arg & 0xffffff);
}

// sceGuDrawArray replacement: append VTYPE/BASE/VADR/PRIM directly (the caller
// issues the stall kick)
static inline void GuDrawArrayDirect(
    int prim, int vtype, int count, const void* vertices) {
    if (vtype) {
        GeCmd(GECMD_VTYPE, (unsigned int)vtype);
    }
    if (vertices) {
        GeCmd(GECMD_BASE, (((unsigned int)vertices) >> 8) & 0xf0000);
        GeCmd(GECMD_VADR, (unsigned int)vertices);
    }
    GeCmd(GECMD_PRIM, ((unsigned int)prim << 16) | (unsigned int)count);
}

// sceGuGetMemory replacement: carve scratch vertex space out of the packet
// list, with a BASE+JUMP pair in front so the GE skips the carved bytes
static inline void* GuGetMemoryDirect(int size) {
    size += 3;
    size += ((unsigned int)(size >> 31)) >> 30;
    size = (size >> 2) << 2;
    unsigned int* orig = gu_list->current;
    unsigned int* next = (unsigned int*)(((unsigned int)orig) + size + 8);
    orig[0] = (GECMD_BASE << 24) | ((((unsigned int)next) >> 8) & 0xf0000);
    orig[1] = (GECMD_JUMP << 24) | (((unsigned int)next) & 0xffffff);
    gu_list->current = next;
    return orig + 2;
}

// ===== pre-baked GE states =====
#define GECMD_TME 0x1E   // texture map enable
#define GECMD_DTE 0x20   // dither enable
#define GECMD_ABE 0x21   // alpha blend enable
#define GECMD_ATE 0x22   // alpha test enable
#define GECMD_BLEND 0xDF // blend equation/factors
#define GECMD_FIXA 0xE0  // blend fixed color A (srcfix)
#define GECMD_FIXB 0xE1  // blend fixed color B (destfix)
#define GECMD_RET 0x0B   // return from CALL

#define SUBLIST_WORDS 8
static unsigned int __attribute__((aligned(16))) sl_blend[5][SUBLIST_WORDS];
static unsigned int __attribute__((aligned(16))) sl_dither[2][SUBLIST_WORDS];
static unsigned int __attribute__((aligned(16))) sl_textured[2][SUBLIST_WORDS];
static bool sublists_built;

static inline unsigned int GeWord(unsigned int cmd, unsigned int arg) {
    return (cmd << 24) | (arg & 0xffffff);
}

// equivalent of sceGuBlendFunc(op, GU_FIX, GU_FIX, afix, bfix) followed by
// sceGuEnable(GU_BLEND)
static void BakeBlend(
    unsigned int* p, int op, unsigned int afix, unsigned int bfix) {
    *p++ = GeWord(GECMD_BLEND, GU_FIX | (GU_FIX << 4) | ((unsigned)op << 8));
    *p++ = GeWord(GECMD_FIXA, afix);
    *p++ = GeWord(GECMD_FIXB, bfix);
    *p++ = GeWord(GECMD_ABE, 1);
    *p++ = GeWord(GECMD_RET, 0);
}

// pre-calculate blend, dither and texture tables
static void BuildStateCmdList(void) {
    if (sublists_built) {
        return;
    }
    BakeBlend(sl_blend[0], GU_ADD, 0x808080, 0x808080); // B/2 + F/2
    BakeBlend(sl_blend[1], GU_ADD, 0xFFFFFF, 0xFFFFFF); // B + F
    BakeBlend(sl_blend[2], GU_REVERSE_SUBTRACT, 0xFFFFFF, 0xFFFFFF); // B - F
    BakeBlend(sl_blend[3], GU_ADD, 0x404040, 0xFFFFFF);              // B + F/4
    sl_blend[4][0] = GeWord(GECMD_ABE, 0);
    sl_blend[4][1] = GeWord(GECMD_RET, 0);
    sl_dither[0][0] = GeWord(GECMD_DTE, 0);
    sl_dither[0][1] = GeWord(GECMD_RET, 0);
    sl_dither[1][0] = GeWord(GECMD_DTE, 1);
    sl_dither[1][1] = GeWord(GECMD_RET, 0);
    sl_textured[0][0] = GeWord(GECMD_TME, 0);
    sl_textured[0][1] = GeWord(GECMD_ATE, 0);
    sl_textured[0][2] = GeWord(GECMD_RET, 0);
    sl_textured[1][0] = GeWord(GECMD_TME, 1);
    sl_textured[1][1] = GeWord(GECMD_ATE, 1);
    sl_textured[1][2] = GeWord(GECMD_RET, 0);
    // ensure dcache is flushed before GE fetch this data
    sceKernelDcacheWritebackAll();
    sublists_built = true;
}

// emit a CALL to a pre-baked sub-list into the current (direct) frame list
static inline void CallSubList(const void* list) {
    unsigned int a = (unsigned int)list;
    GeCmd(GECMD_BASE, (a >> 8) & 0xf0000);
    GeCmd(GECMD_CALL, a);
}

// points the GE draw list at an EDRAM offset (the surface, or a framebuffer),
// skipping the command when nothing changed. Uses sceGuDrawBufferList so the
// gu context keeps describing the real 480x272 framebuffer (sceGuClear and
// the scissor-disable path read those dimensions).
static void SetDrawTarget(int edram_offset, int stride) {
    if (applied_draw_target == edram_offset) {
        return;
    }
    applied_draw_target = edram_offset;
    sceGuDrawBufferList(GU_PSM_5551, (void*)edram_offset, stride);
}

static void UpdateOutputMapping(void) {
    if (aspect_mode == PSYZ_ASPECT_SQUARE) {
        target_w = game_w;
    } else {
        // adjust aspect ratio
        float vref = is_pal ? 288.0f : 240.0f;
        float aspect = (4.0f / 3.0f) * ((float)set_disp_horiz / 256.0f) /
                       ((float)set_disp_vert / vref);
        target_w = (int)(aspect * (float)game_h + 0.5f);
    }

    // renders to a surface in the EDRAM, then re-renders it stretched.
    // bypassed if PSYZ_ASPECT_SQUARE, or if game resolution is 320x240
    use_surface = target_w != game_w && game_w > 0;
    if (use_surface) {
        surface_stride = (game_w + 7) & ~7;
        surface_slot_bytes = surface_stride * game_h * 2;
        int budget = (int)sceGeEdramGetSize() - EDRAM_SURFACE_OFFSET;
        // two slots: the surface is double-buffered for the deferred present
        if (surface_slot_bytes * 2 > budget) {
            // defensive: no realistic mode is this large. Present 1:1 instead.
            use_surface = false;
            target_w = game_w;
            if (!warned_surface_budget) {
                warned_surface_budget = true;
                WARNF("surface too large for EDRAM, presenting 1:1");
            }
        }
    }

    // placement of the presented output on the physical screen
    if (draw_area_rect.w > 0 && draw_area_rect.h > 0) {
        pres_x = draw_area_rect.x;
        pres_y = draw_area_rect.y;
    } else {
        pres_x = (PSP_SCREEN_W - target_w) / 2; // negative -> crop both sides
        pres_y = (PSP_SCREEN_H - game_h) / 2;   // 240 -> +16; 480 -> -104 crop
    }
    // presented rect clamped to the screen (present scissor and frame capture)
    int rx = pres_x, ry = pres_y, rw = target_w, rh = game_h;
    if (draw_area_rect.w > 0 && draw_area_rect.h > 0) {
        rw = draw_area_rect.w;
        rh = draw_area_rect.h;
    }
    int x0 = rx < 0 ? 0 : rx;
    int y0 = ry < 0 ? 0 : ry;
    int x1 = rx + rw > PSP_SCREEN_W ? PSP_SCREEN_W : rx + rw;
    int y1 = ry + rh > PSP_SCREEN_H ? PSP_SCREEN_H : ry + rh;
    out_x = x0;
    out_y = y0;
    out_w = x1 > x0 ? x1 - x0 : 0;
    out_h = y1 > y0 ? y1 - y0 : 0;
    map_ofs_x = use_surface ? 0 : pres_x;
    map_ofs_y = use_surface ? 0 : pres_y;

    // flush if display mode changes mid-frame
    if (is_init) {
        FlushBatch();
        if (use_surface) {
            SetDrawTarget(EdramSurfaceOffset(surface_idx), surface_stride);
        } else {
            SetDrawTarget(
                cur_draw_fb ? EDRAM_DISP_OFFSET : EDRAM_DRAW_OFFSET, FB_STRIDE);
        }
    }
}

typedef struct {
    unsigned int c;
    short x, y, z;
} CVert; // GU_COLOR_8888 | GU_VERTEX_16BIT

typedef struct {
    short u, v;
    unsigned int c;
    short x, y, z;
} BVert; // GU_TEXTURE_16BIT | GU_COLOR_8888 | GU_VERTEX_16BIT

// ===== primitive batching =====
#define MAX_VERTEX_COUNT 12288
static BVert __attribute__((aligned(64))) g_vring[2][MAX_VERTEX_COUNT];
static int vring_used;  // vertices produced this frame
static int batch_start; // first vertex of the pending (unflushed) batch
static bool warned_vring_budget;

// the GE primitive of the pending batch: GU_TRIANGLES (3 verts each) or
// GU_SPRITES (2 verts each, used for axis-aligned PS1 TILE/SPRT rectangles).
// A batch holds one primitive type; switching type flushes first.
static int batch_prim = GU_TRIANGLES;

static void FlushBatch(void) {
    int n = vring_used - batch_start;
    if (n == 0) {
        return;
    }
    BVert* first = &g_vring[dlist_idx][batch_start];
    batch_start = vring_used;
    sceKernelDcacheWritebackRange(first, n * sizeof(BVert));
    GuDrawArrayDirect(
        batch_prim,
        GU_TEXTURE_16BIT | GU_COLOR_8888 | GU_VERTEX_16BIT | GU_TRANSFORM_2D, n,
        first);
    // kick the GE so it consumes this batch concurrently with the CPU building
    // the next one, instead of waiting for sceGuFinish. This is only possible
    // thanks to the use of PS1 double-buffer used by default in games.
    sceGeListUpdateStallAddr(ge_list_executed[0], gu_list->current);
}

static void ScissorFull(void) {
    if (use_surface) {
        sceGuScissor(0, 0, surface_stride, game_h);
    } else {
        sceGuScissor(0, 0, PSP_SCREEN_W, PSP_SCREEN_H);
    }
}

static void UpdateScissor(void) {
    if (!is_init) {
        return;
    }
    FlushBatch();
    // drawing area relative to the frame, in render-target coordinates (1:1)
    int x0 = map_ofs_x + draw_area_start.x - fb_origin.x;
    int y0 = map_ofs_y + draw_area_start.y - fb_origin.y;
    int x1 = map_ofs_x + draw_area_end.x + 1 - fb_origin.x;
    int y1 = map_ofs_y + draw_area_end.y + 1 - fb_origin.y;
    // clamp to the render target: the surface (0..game) or the presented rect
    int lo_x = use_surface ? 0 : out_x;
    int lo_y = use_surface ? 0 : out_y;
    int hi_x = use_surface ? game_w : out_x + out_w;
    int hi_y = use_surface ? game_h : out_y + out_h;
    if (x0 < lo_x) {
        x0 = lo_x;
    }
    if (y0 < lo_y) {
        y0 = lo_y;
    }
    if (x1 > hi_x) {
        x1 = hi_x;
    }
    if (y1 > hi_y) {
        y1 = hi_y;
    }
    if (x0 >= x1 || y0 >= y1) {
        sceGuScissor(0, 0, 0, 0);
    } else {
        sceGuScissor(x0, y0, x1 - x0, y1 - y0);
    }
}

// ===== VRAM access =====
static inline u16* VramTileRow(int tile, int y) {
    return &g_vram[tile][(y & (VRAM_TILE_H - 1)) << 6];
}

static void MarkDirtyTiles(int x, int y, int w, int h) {
    int tx0 = x >> 6;
    int tx1 = (x + w - 1) >> 6;
    int ty0 = y >> 8;
    int ty1 = (y + h - 1) >> 8;
    for (int ty = ty0; ty <= ty1; ty++) {
        for (int tx = tx0; tx <= tx1; tx++) {
            tile_dirty[tx + ty * VRAM_TILES_X] = true;
        }
    }
}

// writes a single line to the shadowed VRAM
// TODO this can be simplified, as PS1 VRAM should always be X bytes aligned
static void VramLineWrite(int x, int y, const u16* src, int w) {
    while (w > 0) {
        int tile = (x >> 6) + (y >> 8) * VRAM_TILES_X;
        int tx = x & (VRAM_TILE_W - 1);
        int n = VRAM_TILE_W - tx;
        if (n > w) {
            n = w;
        }
        memcpy(VramTileRow(tile, y) + tx, src, n * sizeof(u16));
        src += n;
        x += n;
        w -= n;
    }
}

// reads a single line to the shadowed VRAM
// TODO this can be simplified, as PS1 VRAM should always be X bytes aligned
static void VramLineRead(int x, int y, u16* dst, int w) {
    while (w > 0) {
        int tile = (x >> 6) + (y >> 8) * VRAM_TILES_X;
        int tx = x & (VRAM_TILE_W - 1);
        int n = VRAM_TILE_W - tx;
        if (n > w) {
            n = w;
        }
        memcpy(dst, VramTileRow(tile, y) + tx, n * sizeof(u16));
        dst += n;
        x += n;
        w -= n;
    }
}

// fills line with the specified color
static inline void LineFill(u16* dst, u16 c, int w) {
    int i = 0;
    if (((uintptr_t)dst & 2) && i < w) {
        dst[i++] = c;
    }
    // copy 32-bit chunks instead of 16-bit, marginally faster
    u32 c2 = ((u32)c << 16) | c;
    u32* dst32 = (u32*)(dst + i);
    for (; i + 1 < w; i += 2) {
        *dst32++ = c2;
    }
    if (i < w) {
        dst[i] = c;
    }
}

// fills a whole VRAM rect with color c, iterating one tile column at a time so
// all rows of a column are written contiguously: better cache locality than a
// per-row walk that hops between tiles every 64 pixels
static void VramRectFill(int x, int y, int w, int h, u16 c) {
    int xe = x + w;
    for (int cx = x; cx < xe;) {
        int tx = cx & (VRAM_TILE_W - 1);
        int n = VRAM_TILE_W - tx;
        if (n > xe - cx) {
            n = xe - cx;
        }
        for (int cy = y; cy < y + h;) {
            int tile = (cx >> 6) + (cy >> 8) * VRAM_TILES_X;
            int ty = cy & (VRAM_TILE_H - 1);
            int rows = VRAM_TILE_H - ty;
            if (rows > y + h - cy) {
                rows = y + h - cy;
            }
            u16* base = &g_vram[tile][(ty << 6) + tx];
            for (int r = 0; r < rows; r++) {
                LineFill(base + (r << 6), c, n);
            }
            cy += rows;
        }
        cx += n;
    }
}

// sceGeEdramGetAddr() has a fixed return value, call it once and store result.
static u8* edram_base;

static void* EdramTile(int tile) {
    return edram_base + EDRAM_TILES_OFFSET + tile * VRAM_TILE_BYTES;
}

static int EdramSurfaceOffset(int idx) {
    return EDRAM_SURFACE_OFFSET + idx * surface_slot_bytes;
}

static void* EdramSurface(int idx) {
    return edram_base + EdramSurfaceOffset(idx);
}

static void UploadTileIfDirty(int tile) {
    if (!tile_dirty[tile]) {
        return;
    }
    WaitPrevFrameGpu();
    tile_dirty[tile] = false;
    // don't use sceGuCopyImage, as it's asynchronous and it could start
    // uploading while EdramTile(tile) gets updates somewhere else.
    // I did not measure the iimpact of not using DMA here.
    void* dst = EdramTile(tile);
    memcpy(dst, g_vram[tile], VRAM_TILE_BYTES);
    sceKernelDcacheWritebackRange(dst, VRAM_TILE_BYTES);
}

// ===== texture page and CLUT caches =====
typedef struct {
    bool valid;
    bool dirty;
    u8 page; // 0-31: VRAM cell (page & 0xF) * 64 halfwords, (page >> 4) * 256
    u8 bpp;  // 1=8bpp, 2=16bpp (4bpp pages bind the tile mirror instead)
    u32 last_used;
} PageEntry;

typedef struct {
    bool valid;
    bool dirty;
    u16 clut;
    u8 bpp; // 0=4bpp (16 colors), 1=8bpp (256 colors)
    u32 last_used;
} ClutEntry;

#define PAGECACHE_MAX 8
#define CLUTCACHE_MAX 64
static PageEntry page_cache[PAGECACHE_MAX];
static ClutEntry clut_cache[CLUTCACHE_MAX];
// one 256x256 16bpp page per slot (the worst case, 8bpp uses half); static
// so the deferred GE reads always see stable, 64-byte aligned storage
static u16 __attribute__((aligned(64))) page_bufs[PAGECACHE_MAX][256 * 256];
#define PageBuf(e) (page_bufs[(int)((e) - page_cache)])

// each CLUT is converted three ways to feed the GE alpha test the PS1 STP-bit
// semantics: opaque (only texel 0x0000 transparent), then the two passes of a
// semi-transparent primitive, cutting the STP=1 and STP=0 texels respectively
enum {
    // opaque primitives; only 0x0000 is transparent
    CLUT_PLAIN,

    // pass 1 of a semi-transparent primitive; texels with STP=1 are cut so
    // only the always-opaque ones are written
    CLUT_OPAQUE_PASS,

    // pass 2; only the STP=1 texels blend, the rest are cut
    CLUT_BLEND_PASS,

    // number of variants
    NUM_CLUTS,
};

static u16 __attribute__((aligned(64))) g_clut[CLUTCACHE_MAX][NUM_CLUTS][256];
static u32 cache_clock;

static const void* bound_tex;
static int bound_tex_psm = -1;
static const void* bound_clut;
static bool tex_memo_valid;
static u16 tex_memo_tpage, tex_memo_clut;
static int tex_memo_variant, tex_memo_bpp;

static void PageRect(const PageEntry* e, int* x, int* y, int* w, int* h) {
    *x = (e->page & 0xF) * 64;
    *y = (e->page >> 4) * 256;
    *w = 64 << e->bpp; // halfwords: 128 at 8bpp, 256 at 16bpp
    *h = 256;
}

static void InvalidateVram(int x, int y, int w, int h) {
    // a VRAM write may dirty the page/CLUT the memo points at; drop it
    tex_memo_valid = false;
    for (int i = 0; i < PAGECACHE_MAX; i++) {
        PageEntry* e = &page_cache[i];
        if (!e->valid || e->dirty) {
            continue;
        }
        int tx, ty, tw, th;
        PageRect(e, &tx, &ty, &tw, &th);
        if (x < tx + tw && x + w > tx && y < ty + th && y + h > ty) {
            e->dirty = true;
        }
    }
    for (int i = 0; i < CLUTCACHE_MAX; i++) {
        ClutEntry* e = &clut_cache[i];
        if (!e->valid || e->dirty) {
            continue;
        }
        int cx = (e->clut & 0x3F) * 16;
        int cy = (e->clut >> 6) & 0x1FF;
        int cw = e->bpp ? 256 : 16;
        if (x < cx + cw && x + w > cx && y <= cy && y + h > cy) {
            e->dirty = true;
        }
    }
}

// converts one 8bpp or 16bpp PS1 texture page from the VRAM shadow into the
// entry's main RAM buffer; 16bpp gets the transparency fixup (color 0x0000
// stays fully transparent, everything else becomes opaque)
static void AssemblePage(PageEntry* e) {
    // PageBuf is reused in place, so no frame may still be sampling it
    WaitPrevFrameGpu();
    int x, y, w, h;
    PageRect(e, &x, &y, &w, &h);
    int vw = w;
    if (x + vw > PSX_VRAM_W) {
        vw = PSX_VRAM_W - x; // pages at the right edge of VRAM
    }
    u16 row[256];
    u16* buf = PageBuf(e);
    for (int i = 0; i < h; i++) {
        VramLineRead(x, y + i, row, vw);
        if (e->bpp == 2) {
            u16* dst = buf + i * 256;
            for (int j = 0; j < vw; j++) {
                u16 c = row[j];
                dst[j] = c ? (u16)(c | 0x8000) : 0;
            }
            if (vw < w) {
                memset(dst + vw, 0, (w - vw) * 2);
            }
        } else {
            memcpy((u8*)buf + i * 256, row, vw * 2);
            if (vw < w) {
                memset((u8*)buf + i * 256 + vw * 2, 0, (w - vw) * 2);
            }
        }
    }
    sceKernelDcacheWritebackRange(buf, 256 * (e->bpp == 2 ? 512 : 256));
}

static PageEntry* FindOrCreatePage(u8 page, u8 bpp) {
    PageEntry* lru = NULL;
    PageEntry* found = NULL;
    for (int i = 0; i < PAGECACHE_MAX; i++) {
        PageEntry* e = &page_cache[i];
        if (e->valid && e->page == page && e->bpp == bpp) {
            found = e;
            break;
        }
        if (!e->valid) {
            if (!lru || lru->valid) {
                lru = e;
            }
        } else if (!lru || (lru->valid && e->last_used < lru->last_used)) {
            lru = e;
        }
    }
    if (!found) {
        found = lru;
        found->valid = true;
        found->dirty = true;
        found->page = page;
        found->bpp = bpp;
    }
    if (found->dirty) {
        AssemblePage(found);
        found->dirty = false;
    }
    found->last_used = ++cache_clock;
    return found;
}

static ClutEntry* FindOrCreateClut(u16 clut, u8 bpp) {
    ClutEntry* lru = NULL;
    ClutEntry* found = NULL;
    for (int i = 0; i < CLUTCACHE_MAX; i++) {
        ClutEntry* e = &clut_cache[i];
        if (e->valid && e->clut == clut && e->bpp == bpp) {
            found = e;
            break;
        }
        if (!e->valid) {
            if (!lru || lru->valid) {
                lru = e;
            }
        } else if (!lru || (lru->valid && e->last_used < lru->last_used)) {
            lru = e;
        }
    }
    if (!found) {
        found = lru;
        found->valid = true;
        found->dirty = true;
        found->clut = clut;
        found->bpp = bpp;
    }
    if (found->dirty) {
        WaitPrevFrameGpu();
        // texel 0x0000 is always fully transparent; the STP bit (0x8000)
        // selects which texels take part in semi-transparent blending. The
        // alpha bit of each variant feeds the GE alpha test.
        int idx = (int)(found - clut_cache);
        int cx = (clut & 0x3F) * 16;
        int cy = (clut >> 6) & 0x1FF;
        int colors = bpp ? 256 : 16;
        if (cx + colors > PSX_VRAM_W) {
            colors = PSX_VRAM_W - cx;
        }
        u16 row[256];
        VramLineRead(cx, cy, row, colors);
        for (int i = 0; i < colors; i++) {
            u16 c = row[i];
            u16 opaque = (u16)(c | 0x8000);
            u16 cut = (u16)(c & 0x7FFF);
            g_clut[idx][CLUT_PLAIN][i] = c ? opaque : 0;
            g_clut[idx][CLUT_OPAQUE_PASS][i] =
                c && !(c & 0x8000) ? opaque : cut;
            g_clut[idx][CLUT_BLEND_PASS][i] = c & 0x8000 ? opaque : cut;
        }
        sceKernelDcacheWritebackRange(g_clut[idx], sizeof(g_clut[0]));
        found->dirty = false;
    }
    found->last_used = ++cache_clock;
    return found;
}

// ===== GE bring-up and frame loop =====
static const ScePspIMatrix4 psx_dither_matrix = {
    {-4, 0, -3, 1},
    {2, -2, 3, -1},
    {-3, 1, -4, 0},
    {3, -1, 2, -2},
};

// TODO consider using sceGuClear instead of rendering a quad
static void ClearTargetBlack(int w, int h) {
    ScissorFull();
    CVert* out = GuGetMemoryDirect(2 * sizeof(CVert));
    out[0].c = 0xFF000000;
    out[0].x = 0;
    out[0].y = 0;
    out[0].z = 0;
    out[1].c = 0xFF000000;
    out[1].x = (short)w;
    out[1].y = (short)h;
    out[1].z = 0;
    GuDrawArrayDirect(
        GU_SPRITES, GU_COLOR_8888 | GU_VERTEX_16BIT | GU_TRANSFORM_2D, 2, out);
}

static void StartFrame(void) {
    fb_origin_locked = false;
    warned_list_budget = false;
    vring_used = 0;
    batch_start = 0;
    batch_prim = GU_TRIANGLES;
    warned_vring_budget = false;
    list_budget_calls = 0;
    list_budget_cached = true;
    cur_blend = -1;
    cur_dither = -1;
    cur_textured = -1;
    bound_tex = NULL;
    bound_tex_psm = -1;
    bound_clut = NULL;
    tex_memo_valid = false;

    cur_draw_fb ^= 1;
    sceGuStart(GU_DIRECT, dlist[dlist_idx]);
    applied_draw_target = EDRAM_DRAW_OFFSET;
    sceGuClearColor(0xFF000000);
    sceGuDisable(GU_BLEND);
    sceGuDisable(GU_DITHER);
    sceGuDisable(GU_TEXTURE_2D);
    if (use_surface) {
        SetDrawTarget(EdramSurfaceOffset(surface_idx), surface_stride);
        ClearTargetBlack(surface_stride, game_h);
    } else {
        SetDrawTarget(FbOffset(cur_draw_fb), FB_STRIDE);
        ScissorFull();
        sceGuClear(GU_COLOR_BUFFER_BIT);
    }
    UpdateScissor();
}

static int Pow2Ceil(int v) {
    int p = 1;
    while (p < v) {
        p <<= 1;
    }
    return p;
}

static void PresentSurface(void) {
    SetDrawTarget(FbOffset(cur_draw_fb), FB_STRIDE);
    ClearTargetBlack(PSP_SCREEN_W, PSP_SCREEN_H); // letterbox/pillarbox
    if (out_w <= 0 || out_h <= 0) {
        return; // draw area fully offscreen
    }
    sceGuScissor(out_x, out_y, out_w, out_h);
    sceGuDisable(GU_BLEND);
    sceGuDisable(GU_DITHER);
    sceGuDisable(GU_ALPHA_TEST);
    sceGuEnable(GU_TEXTURE_2D);
    sceGuTexMode(GU_PSM_5551, 0, 0, 0);
    sceGuTexImage(0, Pow2Ceil(game_w), Pow2Ceil(game_h), surface_stride,
                  EdramSurface(surface_idx));
    BVert* out = GuGetMemoryDirect(2 * sizeof(BVert));
    out[0].u = 0;
    out[0].v = 0;
    out[0].c = 0xFF7F7F7Fu;
    out[0].x = (short)pres_x;
    out[0].y = (short)pres_y;
    out[0].z = 0;
    out[1].u = (short)game_w;
    out[1].v = (short)game_h;
    out[1].c = 0xFF7F7F7Fu;
    out[1].x = (short)(pres_x + target_w);
    out[1].y = (short)(pres_y + game_h);
    out[1].z = 0;
    GuDrawArrayDirect(
        GU_SPRITES,
        GU_TEXTURE_16BIT | GU_COLOR_8888 | GU_VERTEX_16BIT | GU_TRANSFORM_2D, 2,
        out);
}

bool InitPlatform(void) {
    if (is_init) {
        return true;
    }
    edram_base = (u8*)sceGeEdramGetAddr();
    sceGuInit();
    sceGuStart(GU_DIRECT, dlist[0]);
    sceGuDrawBuffer(GU_PSM_5551, (void*)EDRAM_DRAW_OFFSET, FB_STRIDE);
    sceGuDispBuffer(
        PSP_SCREEN_W, PSP_SCREEN_H, (void*)EDRAM_DISP_OFFSET, FB_STRIDE);
    sceGuOffset(2048 - (PSP_SCREEN_W / 2), 2048 - (PSP_SCREEN_H / 2));
    sceGuViewport(2048, 2048, PSP_SCREEN_W, PSP_SCREEN_H);
    sceGuScissor(0, 0, PSP_SCREEN_W, PSP_SCREEN_H);
    sceGuEnable(GU_SCISSOR_TEST);
    sceGuDisable(GU_DEPTH_TEST);
    sceGuDepthMask(GU_TRUE);
    sceGuShadeModel(GU_SMOOTH);
    sceGuDisable(GU_CULL_FACE);
    sceGuDisable(GU_TEXTURE_2D);
    sceGuDisable(GU_BLEND);
    sceGuSetDither(&psx_dither_matrix);
    sceGuDisable(GU_DITHER);
    sceGuEnable(GU_FRAGMENT_2X); // use 0x7F is 1.0 of brightness
    sceGuTexFunc(GU_TFX_MODULATE, GU_TCC_RGBA);
    sceGuTexFilter(GU_NEAREST, GU_NEAREST);
    sceGuTexWrap(GU_REPEAT, GU_REPEAT);  // UVs wrap within a 256x256 page
    sceGuAlphaFunc(GU_GREATER, 0, 0xFF); // discard 0x0000 colors
    sceGuDisable(GU_ALPHA_TEST);
    sceGuFinish();
    sceGuSync(0, 0);
    BuildStateCmdList();
    sceDisplayWaitVblankStart();
    sceGuDisplay(GU_TRUE);

    is_init = true;
    UpdateOutputMapping();
    StartFrame();
    return true;
}

void ResetPlatform(void) {
    if (is_init) {
        sceGuTerm();
        is_init = false;
    }
    cur_tpage = 0;
    internal_res = 1;
}

// ===== vsync, pacing, video API =====

// wait for the previous frame's GE list before its buffers get reused
// cheap to use, but ensure buffers do not get used in two places at once
static void WaitPrevFrameGpu(void) {
    if (!prev_frame_pending) {
        return;
    }
    prev_frame_pending = false;
    // never sceGeDrawSync, as it would also wait on the still-open list,
    // deadlocking
    sceGeListSync(prev_frame_list_id, 0);
}

// EDRAM byte offset of framebuffer index
static int FbOffset(int idx) {
    return idx ? EDRAM_DISP_OFFSET : EDRAM_DRAW_OFFSET;
}

// point the display at show_fb, which has the frame ready to show on screen
static void ShowPendingFrame(void) {
    if (!pending_show) {
        return;
    }
    pending_show = false;
    // the vblank wait is the only frame pacing the PSP has: the console is
    // fixed at ~59.94Hz, so AUTO/ON/OFF all land on it and only LIMITLESS
    // runs the game as fast as the GE and CPU allow
    if (vsync_mode != PSYZ_VSYNC_LIMITLESS) {
        sceDisplayWaitVblankStart();
    }
    sceDisplaySetFrameBuf(
        (void*)(edram_base + FbOffset(show_fb)), FB_STRIDE,
        PSP_DISPLAY_PIXEL_FORMAT_5551, PSP_DISPLAY_SETBUF_IMMEDIATE);
    sceGuDisplay(display_enabled ? GU_TRUE : GU_FALSE);
    last_vsync = sceDisplayGetVcount();
}

int Psyz_VideoVSync(int mode) {
    if (!is_init && !InitPlatform()) {
        return 0;
    }
    unsigned int cur = sceDisplayGetVcount();
    int ret = (int)((cur - last_vsync) * 1000 / 60); // TODO logic incorrect?
    last_vsync = cur;
    if (mode == 0) {
        WaitPrevFrameGpu();
        ShowPendingFrame();

        FlushBatch(); // emit any pending batched primitives before presenting
        if (use_surface) {
            // stretch this frame's off-screen surface onto its framebuffer
            PresentSurface();
        }
        sceGuFinish(); // sync current GE list
        prev_frame_list_id = ge_list_executed[0];
        prev_frame_pending = true;
        show_fb = cur_draw_fb;
        pending_show = true;
        dlist_idx ^= 1;
        surface_idx ^= 1;
        StartFrame();
    }
    return ret;
}

int Psyz_VideoSetVsyncMode(PsyzVsyncMode mode) {
    switch (mode) {
    case PSYZ_VSYNC_AUTO:
    case PSYZ_VSYNC_ON:
    case PSYZ_VSYNC_OFF:
    case PSYZ_VSYNC_LIMITLESS:
        vsync_mode = mode;
        return 0;
    default:
        return -1;
    }
}

int Psyz_VideoSetDitheringMode(PsyzDitherMode mode) {
    switch (mode) {
    case PSYZ_DITHER_AUTO:
    case PSYZ_DITHER_OFF:
        dither_mode = mode;
        return 0;
    default:
        return -1;
    }
}

int Psyz_VideoSetAspectMode(PsyzAspectMode mode) {
    switch (mode) {
    case PSYZ_ASPECT_DISPLAY:
    case PSYZ_ASPECT_SQUARE:
        aspect_mode = mode;
        UpdateOutputMapping();
        UpdateScissor();
        return 0;
    default:
        return -1;
    }
}

PsyzSize Psyz_VideoGetDisplaySize(void) {
    PsyzSize s = {PSP_SCREEN_W, PSP_SCREEN_H};
    if (aspect_mode == PSYZ_ASPECT_DISPLAY) {
        // largest 4:3 area at 272 tall
        // keeps the output pixel-perfect while preserving the aspect ratio
        s.w = 362;
    }
    return s;
}

void Psyz_VideoSetDrawArea(PsyzRect rect) {
    draw_area_rect = rect;
    UpdateOutputMapping();
    UpdateScissor();
}

int Psyz_VideoStats(PsyzVideoStats* stats) {
    if (!stats || !is_init) {
        return -1;
    }
    memset(stats, 0, sizeof(*stats));
    stats->target_frame_time_us = 1000000.0 / 60.0;
    stats->total_frames = sceDisplayGetVcount();
    stats->using_driver_vsync = vsync_mode != PSYZ_VSYNC_LIMITLESS;
    return 0;
}

unsigned char* Psyz_VideoAllocCapturedFrame(int* w, int* h) {
    if (!is_init && !InitPlatform()) {
        return NULL;
    }
    WaitPrevFrameGpu();
    int fb_idx = show_fb >= 0 ? show_fb : cur_draw_fb;
    int bufw = FB_STRIDE;
    const u16* fb =
        (const u16*)(0x40000000 | (uintptr_t)(edram_base + FbOffset(fb_idx)));
    unsigned char* out = malloc(out_w * out_h * 3);
    if (!out) {
        return NULL;
    }
    unsigned char* p = out;
    for (int y = 0; y < out_h; y++) {
        const u16* row = &fb[(out_y + y) * bufw + out_x];
        for (int x = 0; x < out_w; x++) {
            u16 c = row[x];
            u8 r5 = c & 0x1F;
            u8 g5 = (c >> 5) & 0x1F;
            u8 b5 = (c >> 10) & 0x1F;
            *p++ = (u8)((r5 << 3) | (r5 >> 2));
            *p++ = (u8)((g5 << 3) | (g5 >> 2));
            *p++ = (u8)((b5 << 3) | (b5 >> 2));
        }
    }
    if (w) {
        *w = out_w;
    }
    if (h) {
        *h = out_h;
    }
    return out;
}

// ===== display state =====

void Draw_Reset(void) {
    if (!is_init && !InitPlatform()) {
        return;
    }
    cur_tpage = 0;
    env_dither = false;
    draw_offset.x = 0;
    draw_offset.y = 0;
    draw_area_start.x = 0;
    draw_area_start.y = 0;
    draw_area_end.x = game_w - 1;
    draw_area_end.y = game_h - 1;
    fb_origin.x = 0;
    fb_origin.y = 0;
    disp_origin_count = 0;
    display_enabled = true;
    UpdateOutputMapping();
    UpdateScissor();
}

void Draw_DisplayEnable(unsigned int on) { display_enabled = on != 0; }

void Draw_DisplayArea(unsigned int x, unsigned int y) {
    // remember the framebuffer locations (typically two, alternating)
    for (int i = 0; i < disp_origin_count; i++) {
        if (disp_origins[i].x == (int)x && disp_origins[i].y == (int)y) {
            return;
        }
    }
    if (disp_origin_count < 2) {
        disp_origins[disp_origin_count].x = (int)x;
        disp_origins[disp_origin_count].y = (int)y;
        disp_origin_count++;
    } else {
        // a third origin: the game moved its framebuffers, forget the oldest
        disp_origins[0] = disp_origins[1];
        disp_origins[1].x = (int)x;
        disp_origins[1].y = (int)y;
    }
}

void Draw_DisplayHorizontalRange(unsigned int start, unsigned int end) {
    set_disp_horiz = (int)(end - start) / 10;
    UpdateOutputMapping();
    UpdateScissor();
}

void Draw_DisplayVerticalRange(unsigned int start, int unsigned end) {
    set_disp_vert = (int)(end - start);
    UpdateOutputMapping();
    UpdateScissor();
}

void Draw_SetDisplayMode(DisplayMode* mode) {
    if (mode->reversed) {
        WARNF("reverse mode not supported");
    }
    if (mode->rgb24) {
        LOG_ONCE("24-bit display mode not supported");
    }
    is_pal = mode->pal;
    if (mode->horizontal_resolution_368) {
        game_w = 368;
    } else {
        switch (mode->horizontal_resolution) {
        case 0:
            game_w = 256;
            break;
        case 1:
            game_w = 320;
            break;
        case 2:
            game_w = 512;
            break;
        case 3:
            game_w = 640;
            break;
        }
    }
    game_h = mode->vertical_resolution ? 480 : 240;
    UpdateOutputMapping();
    UpdateScissor();
}

void Draw_PutDispEnv(DISPENV* disp) { (void)disp; }

void Draw_SetTexpageMode(ParamDrawTexpageMode* p) {
    cur_tpage = *(u_short*)p & 0x1FF;
    env_dither = p->dither != 0;
    if (p->tex_flip_x || p->tex_flip_y) {
        LOG_ONCE("texture page flipping not implemented");
    }
}

void Draw_SetTextureWindow(unsigned int mask_x, unsigned int mask_y,
                           unsigned int off_x, unsigned int off_y) {
    if (off_x > 0 || off_y > 0 || mask_x != 0xFFFFFFFF ||
        mask_y != 0xFFFFFFFF) {
        NOT_IMPLEMENTED;
    }
}

void Draw_SetAreaStart(int x, int y) {
    draw_area_start.x = x;
    draw_area_start.y = y;
    if (!fb_origin_locked) {
        int origin_x = x, origin_y = y;
        bool is_framebuffer = disp_origin_count == 0;
        for (int i = 0; i < disp_origin_count; i++) {
            if (x >= disp_origins[i].x && x < disp_origins[i].x + game_w &&
                y >= disp_origins[i].y && y < disp_origins[i].y + game_h) {
                origin_x = disp_origins[i].x;
                origin_y = disp_origins[i].y;
                is_framebuffer = true;
                break;
            }
        }
        if (is_framebuffer) {
            fb_origin_locked = true;
            fb_origin.x = origin_x;
            fb_origin.y = origin_y;
        }
    }
    UpdateScissor();
}

void Draw_SetAreaEnd(int x, int y) {
    draw_area_end.x = x;
    draw_area_end.y = y;
    UpdateScissor();
}

void Draw_SetOffset(int x, int y) {
    x = x % PSX_VRAM_W;
    y = y % PSX_VRAM_H;
    if (x < 0) {
        x += PSX_VRAM_W;
    }
    if (y < 0) {
        y += PSX_VRAM_H;
    }
    draw_offset.x = x;
    draw_offset.y = y;
}

void Draw_SetMask(int bit0, int bit1) {
    if (bit0 || bit1) {
        NOT_IMPLEMENTED;
    }
}

// ===== image transfers =====

static void ClampRect(const RECT* in, int* x, int* y, int* w, int* h) {
    *x = in->x & (PSX_VRAM_W - 1);
    *y = in->y & (PSX_VRAM_H - 1);
    *w = in->w;
    *h = in->h;
    if (*x + *w > PSX_VRAM_W) {
        *w = PSX_VRAM_W - *x;
    }
    if (*y + *h > PSX_VRAM_H) {
        *h = PSX_VRAM_H - *y;
    }
}

static void EmitClearQuad(int x, int y, int w, int h, u8 r, u8 g, u8 b);
static void EmitVramRegionQuad(int x, int y, int w, int h);

void Draw_ClearImage(RECT* rect, u_char r, u_char g, u_char b) {
    int x, y, w, h;
    if (rect->w == 0 || rect->h == 0) {
        return;
    }
    if (!is_init && !InitPlatform()) {
        return;
    }
    ClampRect(rect, &x, &y, &w, &h);
    u16 col = (u16)((r >> 3) | ((g >> 3) << 5) | ((b >> 3) << 10));
    VramRectFill(x, y, w, h, col);
    MarkDirtyTiles(x, y, w, h);
    InvalidateVram(x, y, w, h);
    EmitClearQuad(x, y, w, h, r, g, b);
}

void Draw_LoadImage(RECT* rect, u_long* p) {
    int x, y, w, h;
    if (rect->w == 0 || rect->h == 0) {
        return;
    }
    ClampRect(rect, &x, &y, &w, &h);
    const u16* src = (const u16*)p;
    for (int i = 0; i < h; i++) {
        VramLineWrite(x, y + i, src, w);
        src += rect->w;
    }
    MarkDirtyTiles(x, y, w, h);
    InvalidateVram(x, y, w, h);
    EmitVramRegionQuad(x, y, w, h);
}

void Draw_StoreImage(RECT* rect, u_long* p) {
    int x, y, w, h;
    if (rect->w == 0 || rect->h == 0) {
        return;
    }
    ClampRect(rect, &x, &y, &w, &h);
    u16* dst = (u16*)p;
    for (int i = 0; i < h; i++) {
        VramLineRead(x, y + i, dst, w);
        dst += rect->w;
    }
}

void Draw_MoveImage(RECT* rect, unsigned int x, unsigned int y) {
    int sx, sy, w, h;
    if (rect->w == 0 || rect->h == 0) {
        return;
    }
    ClampRect(rect, &sx, &sy, &w, &h);
    int dx = (int)x & (PSX_VRAM_W - 1);
    int dy = (int)y & (PSX_VRAM_H - 1);
    if (dx + w > PSX_VRAM_W) {
        w = PSX_VRAM_W - dx;
    }
    if (dy + h > PSX_VRAM_H) {
        h = PSX_VRAM_H - dy;
    }
    // simulate MoveImage PS1 bug on overlapping destination
    // TODO might not if what's moved comes from DISP/DRAW buffer
    u16 row[PSX_VRAM_W];
    if (dy > sy) {
        for (int i = h - 1; i >= 0; i--) {
            VramLineRead(sx, sy + i, row, w);
            VramLineWrite(dx, dy + i, row, w);
        }
    } else {
        for (int i = 0; i < h; i++) {
            VramLineRead(sx, sy + i, row, w);
            VramLineWrite(dx, dy + i, row, w);
        }
    }
    MarkDirtyTiles(dx, dy, w, h);
    InvalidateVram(dx, dy, w, h);
    EmitVramRegionQuad(dx, dy, w, h);
}

// ===== primitive rendering =====

typedef struct {
    short x, y;
    u16 u, v;
    u8 r, g, b, a; // a: 0x80 = semi-transparent primitive, 0xFF = opaque
} PVert;

static inline void BatchTri(const BVert* vs, int a, int b, int c) {
    if (batch_prim != GU_TRIANGLES) {
        FlushBatch(); // flush different primitives first
        batch_prim = GU_TRIANGLES;
    }
    if (vring_used + 3 > MAX_VERTEX_COUNT) {
        if (!warned_vring_budget) {
            warned_vring_budget = true;
            WARNF("vertex ring exhausted, dropping primitives");
        }
        return;
    }
    BVert* out = &g_vring[dlist_idx][vring_used];
    vring_used += 3;
    out[0] = vs[a];
    out[1] = vs[b];
    out[2] = vs[c];
}

static inline void BatchSprite(const BVert* v0, const BVert* v1) {
    if (batch_prim != GU_SPRITES) {
        FlushBatch(); // pending triangles use a different GE primitive
        batch_prim = GU_SPRITES;
    }
    if (vring_used + 2 > MAX_VERTEX_COUNT) {
        if (!warned_vring_budget) {
            warned_vring_budget = true;
            WARNF("vertex ring exhausted, dropping primitives");
        }
        return;
    }
    BVert* out = &g_vring[dlist_idx][vring_used];
    vring_used += 2;
    out[0] = *v0;
    out[1] = *v1;
}

static bool ListBudgetSlow(void) {
    list_budget_calls = 0;
    // sceGuCheckList returns how much of the active list is used, in words
    if (sceGuCheckList() * 4u > DLIST_WATERMARK) {
        if (!warned_list_budget) {
            warned_list_budget = true;
            WARNF("display list budget exhausted, dropping primitives");
        }
        list_budget_cached = false;
        return false;
    }
    list_budget_cached = true;
    return true;
}

static inline bool ListBudget(void) {
    if (++list_budget_calls < 16 && list_budget_cached) {
        return true;
    }
    return ListBudgetSlow();
}

// the Apply* setters run per primitive and usually find the state unchanged,
// so the compare is inline and the GE work sits in an out-of-line *Slow body
static void ApplyBlendSlow(int want) {
    FlushBatch(); // state change: emit pending prims before it takes effect
    cur_blend = want;
    CallSubList(sl_blend[want < 0 ? 4 : want]);
}

static inline void ApplyBlend(bool semitrans, int abr) {
    int want = semitrans ? abr : -1;
    if (want != cur_blend) {
        ApplyBlendSlow(want);
    }
}

static void ApplyDitherSlow(int v) {
    FlushBatch();
    cur_dither = v;
    CallSubList(sl_dither[v ? 1 : 0]);
}

static inline void ApplyDither(bool want) {
    int v = want && dither_mode == PSYZ_DITHER_AUTO ? 1 : 0;
    if (v != cur_dither) {
        ApplyDitherSlow(v);
    }
}

static inline int ApplyTexture(u16 tpage, u16 clut, int variant) {
    // fast path: texture is already binded with same clut
    // TODO can't PSP bind a clut as a GU command like on PS1, instead of
    // flushing the batch on every clut change? That should improve SOTN port
    if (tex_memo_valid && tpage == tex_memo_tpage && clut == tex_memo_clut &&
        variant == tex_memo_variant) {
        return tex_memo_bpp;
    }

    // bpp==0: 4bpp, GU_PSM_T4, needs 16 color palette
    // bpp==1: 8bpp, GU_PSM_T8, needs 256 color palette
    // bpp==2: 16bpp, GU_PSM_5551, bitmap
    u8 page = tpage & 0x1F;
    u8 bpp = (tpage >> 7) & 3;
    if (bpp > 2) {
        bpp = 2;
    }
    const void* tex;
    int tex_psm;
    if (bpp == 0) {
        int tile = (page & 0xF) + (page >> 4) * VRAM_TILES_X;
        UploadTileIfDirty(tile);
        tex = EdramTile(tile);
        tex_psm = GU_PSM_T4;
    } else {
        PageEntry* e = FindOrCreatePage(page, bpp);
        if (!e) {
            return -1;
        }
        tex = PageBuf(e);
        tex_psm = bpp == 1 ? GU_PSM_T8 : GU_PSM_5551;
    }
    if (tex != bound_tex || tex_psm != bound_tex_psm) {
        FlushBatch(); // texture bind changes: emit prims using the old texture
        sceGuTexMode(tex_psm, 0, 0, 0);
        sceGuTexImage(0, 256, 256, 256, tex);
        bound_tex = tex;
        bound_tex_psm = tex_psm;
    }
    if (bpp != 2) {
        ClutEntry* e = FindOrCreateClut(clut, bpp);
        if (!e) {
            return -1;
        }
        const u16* colors = g_clut[(int)(e - clut_cache)][variant];
        if (colors != bound_clut) {
            FlushBatch(); // CLUT bind changes: emit prims using the old CLUT
            sceGuClutMode(GU_PSM_5551, 0, 0xFF, 0);
            sceGuClutLoad(bpp ? 32 : 2, colors);
            bound_clut = colors;
        }
    }
    tex_memo_valid = true;
    tex_memo_tpage = tpage;
    tex_memo_clut = clut;
    tex_memo_variant = variant;
    tex_memo_bpp = bpp;
    return bpp;
}

// move vertices at the centre of the screen
static inline short MapX(int x) {
    return (short)(x + draw_offset.x - fb_origin.x + map_ofs_x);
}

static inline short MapY(int y) {
    return (short)(y + draw_offset.y - fb_origin.y + map_ofs_y);
}

static inline unsigned int PackColor(u8 r, u8 g, u8 b) {
    return 0xFF000000u | ((unsigned int)b << 16) | ((unsigned int)g << 8) | r;
}

static inline void ApplyTextured(bool on) {
    if ((int)on != cur_textured) {
        FlushBatch();
        cur_textured = on;
        CallSubList(sl_textured[on ? 1 : 0]);
    }
}

static const u8 tex_color_scale[256] = {
    0,   1,   2,   3,   3,   5,   6,   6,   7,   8,   9,   10,  11,  12,  13,
    14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,
    29,  30,  31,  31,  33,  34,  35,  35,  36,  37,  39,  39,  40,  42,  42,
    43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,
    58,  59,  60,  60,  63,  63,  63,  65,  65,  67,  68,  68,  70,  70,  71,
    72,  73,  75,  75,  76,  77,  78,  79,  80,  81,  81,  83,  84,  85,  86,
    87,  88,  89,  90,  91,  92,  93,  93,  94,  96,  97,  97,  98,  99,  100,
    102, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
    116, 117, 118, 119, 121, 121, 121, 121, 127, 127, 127, 127, 127, 129, 130,
    130, 132, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 143, 143, 144,
    145, 146, 147, 148, 149, 150, 151, 151, 152, 153, 156, 156, 157, 158, 159,
    160, 160, 161, 163, 164, 164, 166, 166, 167, 168, 169, 170, 171, 172, 172,
    174, 176, 176, 178, 178, 180, 180, 181, 182, 183, 183, 183, 188, 188, 188,
    188, 190, 191, 192, 192, 194, 194, 195, 196, 197, 199, 199, 200, 202, 203,
    203, 205, 205, 205, 208, 208, 210, 210, 212, 212, 214, 214, 215, 215, 218,
    218, 219, 219, 221, 222, 222, 225, 225, 226, 226, 228, 228, 230, 231, 233,
    233, 233, 235, 235, 236, 236, 238, 239, 245, 245, 245, 245, 245, 245, 245,
    245,
};

static inline u8 ScaleTexChannel(u8 c) { return tex_color_scale[c]; }

static inline unsigned int PackTexColor(u8 r, u8 g, u8 b) {
    return PackColor(
        ScaleTexChannel(r), ScaleTexChannel(g), ScaleTexChannel(b));
}

static void EmitPrim(const PVert* v, int n, u16 tpage, u16 clut, bool textured,
                     bool semitrans, bool is_rect) {
    // TODO ditch textured+semitrans, they're already packed in tpage
    if (!display_enabled || !ListBudget()) {
        return;
    }
    // build the strip-order vertices as BVert (UVs unused when untextured); an
    // is_rect sprite only needs the two opposite corners tv[0] and tv[3]
    BVert tv[4];
    int build[4], build_n;
    // TODO is is_rect going to degradate performance due to the batching?
    // can we alway issue two triangles instead without putting presusre on GU?
    if (is_rect) {
        build[0] = 0;
        build[1] = 3;
        build_n = 2;
    } else {
        build[0] = 0;
        build[1] = 1;
        build[2] = 2;
        build[3] = 3;
        build_n = n;
    }
    int kx = draw_offset.x - fb_origin.x + map_ofs_x;
    int ky = draw_offset.y - fb_origin.y + map_ofs_y;
    for (int k = 0; k < build_n; k++) {
        int i = build[k];
        tv[i].u = textured ? (short)v[i].u : 0;
        tv[i].v = textured ? (short)v[i].v : 0;
        tv[i].c = textured ? PackTexColor(v[i].r, v[i].g, v[i].b)
                           : PackColor(v[i].r, v[i].g, v[i].b);
        tv[i].x = (short)(v[i].x + kx);
        tv[i].y = (short)(v[i].y + ky);
        tv[i].z = 0;
    }

    if (!textured) {
        ApplyTextured(false);
        ApplyBlend(semitrans, (tpage >> 5) & 3);
        ApplyDither(env_dither && can_dither);
        if (is_rect) {
            BatchSprite(&tv[0], &tv[3]);
        } else {
            BatchTri(tv, 0, 1, 2);
            if (n == 4) {
                BatchTri(tv, 2, 1, 3);
            }
        }
        return;
    }

    int variant = semitrans ? CLUT_OPAQUE_PASS : CLUT_PLAIN;
    int bpp = ApplyTexture(tpage, clut, variant);
    if (bpp < 0) {
        // no memory left: draw untextured instead of dropping
        EmitPrim(v, n, tpage, clut, false, semitrans, is_rect);
        return;
    }
    ApplyTextured(true);
    ApplyDither(env_dither && can_dither);

    int passes = semitrans && bpp != 2 ? 2 : 1;
    if (passes == 1) {
        // common case: one pass, batchable
        ApplyBlend(semitrans, (tpage >> 5) & 3);
        if (is_rect) {
            BatchSprite(&tv[0], &tv[3]);
        } else {
            BatchTri(tv, 0, 1, 2);
            if (n == 4) {
                BatchTri(tv, 2, 1, 3);
            }
        }
        return;
    }

    for (int pass = 0; pass < passes; pass++) {
        ApplyTexture(tpage, clut, pass ? CLUT_BLEND_PASS : variant);
        ApplyBlend(pass != 0, (tpage >> 5) & 3);
        FlushBatch(); // ensure any earlier batched prims are emitted first
        BVert* out = GuGetMemoryDirect(n * sizeof(BVert));
        memcpy(out, tv, n * sizeof(BVert));
        GuDrawArrayDirect(n == 3 ? GU_TRIANGLES : GU_TRIANGLE_STRIP,
                          GU_TEXTURE_16BIT | GU_COLOR_8888 | GU_VERTEX_16BIT |
                              GU_TRANSFORM_2D,
                          n, out);
    }
}

static void EmitQuadRing(const PVert* v) {
    static const int order[4] = {0, 1, 3, 2};
    if (!display_enabled || !ListBudget()) {
        return;
    }
    ApplyTextured(false);
    ApplyBlend(v[0].a == 0x80, (cur_tpage >> 5) & 3);
    ApplyDither(env_dither);
    CVert* out = GuGetMemoryDirect(4 * sizeof(CVert));
    for (int i = 0; i < 4; i++) {
        const PVert* s = &v[order[i]];
        out[i].c = PackColor(s->r, s->g, s->b);
        out[i].x = MapX(s->x);
        out[i].y = MapY(s->y);
        out[i].z = 0.0f;
    }
    GuDrawArrayDirect(
        GU_TRIANGLE_STRIP, GU_COLOR_8888 | GU_VERTEX_16BIT | GU_TRANSFORM_2D, 4,
        out);
}

// Show an image transfer that landed in the visible framebuffer. Draw it as
// a quad instead of a CPU transfer to not keep the CPU and BUS busy.
// Should help games like Rayman 1
static void EmitVramRegionQuad(int x, int y, int w, int h) {
    if (!is_init || !display_enabled) {
        return;
    }
    int x0 = x > fb_origin.x ? x : fb_origin.x;
    int y0 = y > fb_origin.y ? y : fb_origin.y;
    int x1 = x + w < fb_origin.x + game_w ? x + w : fb_origin.x + game_w;
    int y1 = y + h < fb_origin.y + game_h ? y + h : fb_origin.y + game_h;
    if (x0 >= x1 || y0 >= y1) {
        return;
    }
    FlushBatch();
    ApplyBlend(false, 0);
    ApplyDither(false);
    sceGuEnable(GU_TEXTURE_2D);
    sceGuDisable(GU_ALPHA_TEST); // disabled, so all pixels will land clean
    cur_textured = -1; // force the next primitive to re-apply its state
    tex_memo_valid = false;
    ScissorFull();
    for (int py = y0; py < y1;) {
        // tile rows start at y 0/256 and are 256 tall
        int tile_y = py & ~0xFF;
        int span_y = (tile_y + 256 < y1 ? tile_y + 256 : y1) - py;
        for (int px = x0; px < x1;) {
            if (!ListBudget()) {
                UpdateScissor();
                return;
            }
            int tile_x = px & ~0x3F;
            int span_x = (tile_x + 64 < x1 ? tile_x + 64 : x1) - px;
            int tile = (tile_x >> 6) + (tile_y >> 8) * VRAM_TILES_X;
            UploadTileIfDirty(tile);
            if (EdramTile(tile) != bound_tex || bound_tex_psm != GU_PSM_5551) {
                sceGuTexMode(GU_PSM_5551, 0, 0, 0);
                sceGuTexImage(0, 64, 256, 64, EdramTile(tile));
                bound_tex = EdramTile(tile);
                bound_tex_psm = GU_PSM_5551;
            }
            BVert* out = GuGetMemoryDirect(2 * sizeof(BVert));
            out[0].u = (short)(px - tile_x);
            out[0].v = (short)(py - tile_y);
            out[0].c = 0xFF7F7F7Fu;
            out[0].x = (short)(px - fb_origin.x + map_ofs_x);
            out[0].y = (short)(py - fb_origin.y + map_ofs_y);
            out[0].z = 0;
            out[1].u = (short)(px - tile_x + span_x);
            out[1].v = (short)(py - tile_y + span_y);
            out[1].c = 0xFF7F7F7Fu;
            out[1].x = (short)(px + span_x - fb_origin.x + map_ofs_x);
            out[1].y = (short)(py + span_y - fb_origin.y + map_ofs_y);
            out[1].z = 0;
            GuDrawArrayDirect(GU_SPRITES,
                              GU_TEXTURE_16BIT | GU_COLOR_8888 |
                                  GU_VERTEX_16BIT | GU_TRANSFORM_2D,
                              2, out);
            px += span_x;
        }
        py += span_y;
    }
    UpdateScissor();
}

// makes a ClearImage covering the visible framebuffer show up on screen as
// an opaque quad; FILL ignores the drawing area, so the scissor is bypassed
static void EmitClearQuad(int x, int y, int w, int h, u8 r, u8 g, u8 b) {
    if (!display_enabled || !ListBudget()) {
        return;
    }
    int x0 = x > fb_origin.x ? x : fb_origin.x;
    int y0 = y > fb_origin.y ? y : fb_origin.y;
    int x1 = x + w < fb_origin.x + game_w ? x + w : fb_origin.x + game_w;
    int y1 = y + h < fb_origin.y + game_h ? y + h : fb_origin.y + game_h;
    if (x0 >= x1 || y0 >= y1) {
        return;
    }
    FlushBatch();
    ApplyTextured(false);
    ApplyBlend(false, 0);
    ApplyDither(false);
    CVert* out = GuGetMemoryDirect(2 * sizeof(CVert));
    out[0].c = PackColor(r, g, b);
    out[0].x = (float)(x0 - fb_origin.x + map_ofs_x);
    out[0].y = (float)(y0 - fb_origin.y + map_ofs_y);
    out[0].z = 0.0f;
    out[1].c = PackColor(r, g, b);
    out[1].x = (float)(x1 - fb_origin.x + map_ofs_x);
    out[1].y = (float)(y1 - fb_origin.y + map_ofs_y);
    out[1].z = 0.0f;
    ScissorFull();
    GuDrawArrayDirect(
        GU_SPRITES, GU_COLOR_8888 | GU_VERTEX_16BIT | GU_TRANSFORM_2D, 2, out);
    UpdateScissor();
}

// ===== PS1 GPU packet parsing, identical to sdl3_gpu.c =====

#define SEMITRANSP 0x02
#define TEXTURED 0x04
#define EXTRA_VERTEX 0x08
#define GOURAUD 0x10
#define TRIANGLE 0x20

// real hardware uses XY coords as signed 11-bit
static short s11(short v) { return (short)(((v & 0x7FF) ^ 1024) - 1024); }

static int writePacket(PVert* v, int code, int n, u_long* packet, u16* pOut) {
    int w;
    if (!n) {
        return 0;
    }
    v->x = s11(((short*)packet)[0]);
    v->y = s11(((short*)packet)[1]);
    packet++;
    n--;
    if (!n) {
        return 1;
    }
    w = 1;
    if (code & TEXTURED) {
        v->u = ((u8*)packet)[0];
        v->v = ((u8*)packet)[1];
        *pOut = ((u16*)packet)[1];
        w++;
        packet++;
        n--;
        if (!n) {
            return w;
        }
    } else {
        *pOut = 0;
    }
    if (code & GOURAUD) {
        v++;
        v->r = ((u8*)packet)[0];
        v->g = ((u8*)packet)[1];
        v->b = ((u8*)packet)[2];
        v->a = code & SEMITRANSP ? 0x80 : 0xFF;
        w++;
    }
    return w;
}

static void FixupFlipUV(PVert* v, int hasFourVertices) {
    bool fix_u = (v[0].x > v[1].x) ^ (v[0].u > v[1].u);
    fix_u |= (v[0].x > v[2].x) ^ (v[0].u > v[2].u);
    if (fix_u) {
        v[0].u++;
        v[1].u++;
        v[2].u++;
        if (hasFourVertices) {
            v[3].u++;
        }
    }

    bool fix_v = (v[0].y > v[1].y) ^ (v[0].v > v[1].v);
    fix_v |= (v[0].y > v[2].y) ^ (v[0].v > v[2].v);
    if (fix_v) {
        v[0].v++;
        v[1].v++;
        v[2].v++;
        if (hasFourVertices) {
            v[3].v++;
        }
    }
}

int Draw_PushPrim(u_long* packets, int max_len) {
    int len = max_len;
    int code = (int)(*packets >> 24) & 0xFF;
    bool isPoly = !(code & 0x40);
    bool isLine = (code & 0x40) && !(code & 0x20);
    bool isTile = (code & 0x40) && (code & 0x20);
    bool isTextured = (code & TEXTURED) != 0;
    bool isGouraud = (code & GOURAUD) != 0;
    bool isShadeTex = !((code & 1) && isTextured && !isLine);
    bool isSemiTrans = (code & SEMITRANSP) != 0;
    u16 tpage = -1, clut = -1, pad2, pad3;
    PVert verts[4] = {0};
    PVert* v = verts;

    if (!is_init && !InitPlatform()) {
        return max_len;
    }

    if (isShadeTex) {
        v->r = (unsigned char)(*packets >> 0);
        v->g = (unsigned char)(*packets >> 8);
        v->b = (unsigned char)(*packets >> 16);
    } else {
        v->r = v->g = v->b = 0x80;
    }
    v->a = isSemiTrans ? 0x80 : 0xFF;
    packets++;
    len--;
    if (isPoly) {
        if (code & TRIANGLE) {
            int wr, nVertices;
            wr = writePacket(v++, code, len, packets, &clut);
            packets += wr;
            len -= wr;
            wr = writePacket(v++, code, len, packets, &tpage);
            packets += wr;
            len -= wr;
            wr = writePacket(v++, code, len, packets, &pad2);
            packets += wr;
            len -= wr;

            if (code & EXTRA_VERTEX) {
                nVertices = 4;
                wr = writePacket(v, code, len, packets, &pad3);
                packets += wr;
                len -= wr;
            } else {
                nVertices = 3;
            }
            // HACK last rgb are not read by writePacket, so we patch the
            // amount
            if (isGouraud) {
                packets--;
                len++;
            }

            if (isTextured) {
                FixupFlipUV(verts, code & EXTRA_VERTEX);
            } else {
                clut = -1;
                tpage = cur_tpage;
            }
            if (!isGouraud || !isShadeTex) {
                for (int i = 1; i < nVertices; i++) {
                    verts[i].r = verts[0].r;
                    verts[i].g = verts[0].g;
                    verts[i].b = verts[0].b;
                    verts[i].a = verts[0].a;
                }
            }
            // POLY_* triangles/quads may be sheared or gouraud-shaded, so
            // they cannot collapse to a GU_SPRITES rect (is_rect=false)
            can_dither = isGouraud || (isTextured && isShadeTex);
            EmitPrim(verts, nVertices, isTextured ? tpage : cur_tpage, clut,
                     isTextured, isSemiTrans, false);
            can_dither = false;
        } else {
            // shouldn't happen on a normal PSX application
            WARNF("code %02X not supported", code);
        }
    } else if (isLine) {
        bool padding = true;
        int nPoints = ((code >> 2) & 3) + 1;
        if (nPoints == 1) {
            padding = false;
            nPoints++; // don't ask, have faith
        }

        // accumulate line points first, then convert them to quads
        short px[4], py[4];
        unsigned char cr[4], cg[4], cb[4], ca[4];
        int parsed = 0;
        cr[0] = verts[0].r;
        cg[0] = verts[0].g;
        cb[0] = verts[0].b;
        ca[0] = verts[0].a;
        for (int i = 0; len > 0 && i < nPoints; i++) {
            px[i] = s11(((short*)packets)[0]);
            py[i] = s11(((short*)packets)[1]);
            packets++;
            len--;
            parsed = i + 1;
            if (len > 0 && i + 1 < nPoints && isGouraud) {
                cr[i + 1] = ((u8*)packets)[0];
                cg[i + 1] = ((u8*)packets)[1];
                cb[i + 1] = ((u8*)packets)[2];
                ca[i + 1] = isSemiTrans ? 0x80 : 0xFF;
                packets++;
                len--;
            }
        }
        if (!isGouraud) {
            for (int i = 1; i < parsed; i++) {
                cr[i] = cr[0];
                cg[i] = cg[0];
                cb[i] = cb[0];
                ca[i] = ca[0];
            }
        }
        if (padding) {
            len--;
        }

        int nSegments = parsed - 1;
        for (int s = 0; s < nSegments; s++) {
            short x0 = px[s];
            short y0 = py[s];
            short x1 = px[s + 1];
            short y1 = py[s + 1];
            int dx = x1 - x0;
            int dy = y1 - y0;

            // decides to thicken horizontally or vertically
            short ox, oy;
            if ((dx < 0 ? -dx : dx) >= (dy < 0 ? -dy : dy)) {
                ox = 0;
                oy = 1;
            } else {
                ox = 1;
                oy = 0;
            }

            PVert q[4];
            q[0].x = x0;
            q[0].y = y0;
            q[1].x = x1;
            q[1].y = y1;
            q[2].x = (short)(x1 + ox);
            q[2].y = (short)(y1 + oy);
            q[3].x = (short)(x0 + ox);
            q[3].y = (short)(y0 + oy);
            q[0].r = cr[s];
            q[0].g = cg[s];
            q[0].b = cb[s];
            q[0].a = ca[s];
            q[3].r = cr[s];
            q[3].g = cg[s];
            q[3].b = cb[s];
            q[3].a = ca[s];
            q[1].r = cr[s + 1];
            q[1].g = cg[s + 1];
            q[1].b = cb[s + 1];
            q[1].a = ca[s + 1];
            q[2].r = cr[s + 1];
            q[2].g = cg[s + 1];
            q[2].b = cb[s + 1];
            q[2].a = ca[s + 1];
            EmitQuadRing(q);
        }
    } else if (isTile) {
        int x, y, w = 0, h = 0, tu = 0, tv = 0;
        x = s11(((short*)packets)[0]);
        y = s11(((short*)packets)[1]);
        packets++;
        len--;
        if (isTextured) {
            tu = ((u8*)packets)[0];
            tv = ((u8*)packets)[1];
            clut = ((s16*)packets)[1];
            tpage = cur_tpage;
            packets++;
            len--;
        } else {
            clut = -1;
            tpage = cur_tpage;
        }
        switch (code & ~3) {
        case 0x60: // TILE
        case 0x64: // SPRT
            w = ((s16*)packets)[0];
            h = ((s16*)packets)[1];
            packets++;
            len--;
            break;
        case 0x68: // TILE_1
        case 0x6C:
            w = 1;
            h = 1;
            break;
        case 0x70: // TILE_8
        case 0x74: // SPRT_8
            w = 8;
            h = 8;
            break;
        case 0x78: // TILE_16
        case 0x7C: // SPRT_16
            w = 16;
            h = 16;
            break;
        default:
            WARNF("unrecognized tile code %02X", code);
            break;
        }
        verts[0].x = (short)(x);
        verts[0].y = (short)(y);
        verts[3].x = (short)(x + w);
        verts[3].y = (short)(y + h);
        if (isTextured) {
            verts[0].u = tu;
            verts[0].v = tv;
            verts[3].u = tu + w;
            verts[3].v = tv + h;
        }
        verts[3].r = verts[0].r;
        verts[3].g = verts[0].g;
        verts[3].b = verts[0].b;
        verts[3].a = verts[0].a;
        EmitPrim(verts, 4, tpage, clut, isTextured, isSemiTrans, true);
    }
    return max_len - len;
}

void Draw_ResetBuffer(void) {
    if (!is_init && !InitPlatform()) {
        return;
    }
}

void Draw_FlushBuffer(void) {
    // geometry commands are emitted to the engine immediately
}

int Draw_ExequeSync(void) { return 0; }
