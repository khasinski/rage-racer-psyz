#include <psyz.h>
#include <libgpu.h>
#include <psyz/log.h>
#include "../draw.h"
#include "../internal.h"

// The GPU is a FIFO: the register reflects what it has consumed by the GPU the
// moment it's queried. The problem with the current PsyZ implementation is
// the GPU emulation is synchronous. GPU packets are accumulated in GPU_Enqueue
// then consumed by Psyz_GpuExeque on demand by the same thread. This means the
// values of GPU_STATUS will be stale, especially when 0xE1xxxxxx gets enqueued
// and then GPU_STATUS gets queried before Psyz_GpuExeque is called. But then
// calling Psyz_GpuExeque implies synchronization, which slows down performance.
static u32 GPU_STATUS = 0;

typedef enum {
    // https://psx-spx.consoledev.net/graphicsprocessingunitgpu/#gpu-versions
    GPU_V0, // 1MB VRAM, retail
    GPU_V1, // 2MB VRAM, arcade
    GPU_V2, // 2MB VRAM, PSone
} GpuVersion;

struct Gpu {
    /* 0x00 */ const char* ver;
    /* 0x04 */ int (*addque)(
        int (*exec)(u_long p1, u_long p2), u_long p1, u_long p2);
    /* 0x08 */ int (*addque2)(
        int (*exec)(u_long p1, u_long p2), u_long p1, int len, u_long p2);
    /* 0x0C */ int (*clr)(RECT* rect, u32 color);
    /* 0x10 */ void (*ctl)(unsigned int);
    /* 0x14 */ int (*cwb)();
    /* 0x18 */ int (*cwc)(u_long p1, u_long p2);
    /* 0x1C */ int (*drs)(u_long p1, u_long p2);
    /* 0x20 */ int (*dws)(u_long p1, u_long p2);
    /* 0x24 */ int (*exeque)();
    /* 0x28 */ int (*getctl)(int);
    /* 0x2C */ void (*otc)(OT_TYPE* ot, s32 n);
    /* 0x30 */ int (*param)(int);
    /* 0x34 */ int (*reset)(int);
    /* 0x38 */ u_long (*status)(void);
    /* 0x3C */ int (*sync)(int mode);
};

u32 get_vram_wh(void);

static void GPU_clear_cache() { NOT_IMPLEMENTED; }

static void GPU_write_image() { NOT_IMPLEMENTED; }

static void GPU_read_image() { NOT_IMPLEMENTED; }

static int queue_len = 0;
static u_long queue_buf[0x4000];
static int trace_scene = -1;
static int trace_timer = -1;

void Psyz_GpuTraceContext(int scene, int timer) {
    trace_scene = scene;
    trace_timer = timer;
}

static void TraceOtNode(unsigned long long chain_index, unsigned node_index,
                        const DR_ENV* packet) {
    static int initialized;
    static int enabled;
    static int has_scene;
    static int has_timer;
    static int wanted_scene;
    static int wanted_timer;
    if (!initialized) {
        const char* scene_text = getenv("RAGE_GPU_GP0_TRACE_SCENE");
        const char* timer_text = getenv("RAGE_GPU_GP0_TRACE_TIMER");
        enabled = getenv("RAGE_GPU_OT_TRACE") != NULL;
        has_scene = scene_text != NULL;
        has_timer = timer_text != NULL;
        if (scene_text) wanted_scene = (int)strtol(scene_text, NULL, 0);
        if (timer_text) wanted_timer = (int)strtol(timer_text, NULL, 0);
        initialized = 1;
    }
    if (!enabled || (has_scene && trace_scene != wanted_scene) ||
        (has_timer && trace_timer != wanted_timer)) return;
    if (isendprim(packet)) {
        fprintf(stderr,
                "gp0-node chain=%llu node=%u address=%p next=end words=%u\n",
                chain_index, node_index, (const void*)packet, getlen(packet));
    } else {
        fprintf(stderr,
                "gp0-node chain=%llu node=%u address=%p next=%p words=%u\n",
                chain_index, node_index, (const void*)packet,
                (void*)nextPrim(packet), getlen(packet));
    }
}

/*
 * Host ordering-table links are pointer-sized, but the GPU command stream is
 * always a dense sequence of 32-bit GP0 words.  Primitive structs place their
 * packed payload immediately after O_TAG; drawing-environment packets retain
 * u_long command slots for source compatibility.  Normalize both layouts at
 * this boundary so every renderer and diagnostic consumes the same format.
 */
static int CanonicalizePacket(const DR_ENV* packet, u_long* output,
                              int output_words) {
    int length = getlen(packet);
    int code;
    int i;

    if (length < 0 || length > output_words) return -1;
    if (length == 0) return 0;

    code = getcode(packet) & ~3;
    if (code >= 0x20 && code < 0x80) {
        const u32* packed = (const u32*)packet->code;
        for (i = 0; i < length; i++) output[i] = packed[i];
    } else {
        for (i = 0; i < length; i++) output[i] = packet->code[i] & 0xFFFFFFFFu;
    }
    return length;
}

static void TraceCanonicalPacket(unsigned long long chain_index,
                                 unsigned packet_index, unsigned code,
                                 const DR_ENV* packet,
                                 const u_long* words, int length) {
    static int initialized;
    static int enabled;
    static int has_frame;
    static int has_scene;
    static int has_timer;
    static unsigned long long wanted_frame;
    static int wanted_scene;
    static int wanted_timer;
    PsyzVideoStats stats;
    int i;

    if (!initialized) {
        const char* frame_text = getenv("RAGE_GPU_GP0_TRACE_FRAME");
        const char* scene_text = getenv("RAGE_GPU_GP0_TRACE_SCENE");
        const char* timer_text = getenv("RAGE_GPU_GP0_TRACE_TIMER");
        enabled = getenv("RAGE_GPU_GP0_TRACE") != NULL;
        has_frame = frame_text != NULL;
        has_scene = scene_text != NULL;
        has_timer = timer_text != NULL;
        if (frame_text) wanted_frame = strtoull(frame_text, NULL, 0);
        if (scene_text) wanted_scene = (int)strtol(scene_text, NULL, 0);
        if (timer_text) wanted_timer = (int)strtol(timer_text, NULL, 0);
        initialized = 1;
    }
    if (!enabled || length == 0) return;
    if (Psyz_VideoStats(&stats) != 0) stats.total_frames = 0;
    if (has_frame && stats.total_frames != wanted_frame) return;
    if (has_scene && trace_scene != wanted_scene) return;
    if (has_timer && trace_timer != wanted_timer) return;

    fprintf(stderr,
            "gp0-packet frame=%llu scene=%d timer=%d chain=%llu packet=%u "
            "address=%p code=%02x length=%d "
            "words=",
            stats.total_frames, trace_scene, trace_timer,
            chain_index, packet_index, (const void *)packet, code, length);
    for (i = 0; i < length; i++) {
        fprintf(stderr, "%s%08x", i ? "," : "", (unsigned)words[i]);
    }
    fputc('\n', stderr);
}

static void DispatchPackets(u_long* buf, int len) {
    RECT rect;
    unsigned int x, y;
    for (int i = 0; i < len; i++) {
        u_long op = buf[i];
        int code = (int)(buf[i] >> 24) & 0xFF;
        // https://psx-spx.consoledev.net/graphicsprocessingunitgpu/#gpu-render-polygon-commands
        switch (code) {
        case 0xFF:
            /* Native OT links are pointer-sized.  The flattened 64-bit walk
             * can retain the terminal link word after the last packet; it is
             * a DMA-chain terminator, not a GP0 command. */
            if (i == len - 1)
                break;
            WARNF("unsupported command %02X", code);
            break;
        case 0x00:
            // empty?!
            break;
        case 0x01: // clear cache
            GPU_clear_cache();
            break;
        case 0x02: // frame buffer rectangle draw
            rect.x = (short)(buf[i + 1] & 0xFFFF);
            rect.y = (short)((buf[i + 1] >> 16) & 0xFFFF);
            rect.w = (short)(buf[i + 2] & 0xFFFF);
            rect.h = (short)((buf[i + 2] >> 16) & 0xFFFF);
            Draw_ClearImage(
                &rect, (u_char)(op & 0xFF), (u_char)((op >> 8) & 0xFF),
                (u_char)((op >> 16) & 0xFF));
            i += 2;
            break;
        case 0x80: // move image
            rect.x = (short)(buf[i + 1] & 0xFFFF);
            rect.y = (short)((buf[i + 1] >> 16) & 0xFFFF);
            rect.w = (short)(buf[i + 3] & 0xFFFF);
            rect.h = (short)((buf[i + 3] >> 16) & 0xFFFF);
            x = (short)(buf[i + 2] & 0xFFFF);
            y = (short)((buf[i + 2] >> 16) & 0xFFFF);
            Draw_MoveImage(&rect, x, y);
            i += 3;
            break;
        case 0xA0: // write image
            GPU_write_image();
            break;
        case 0xC0: // read image
            GPU_read_image();
            break;
        case 0xE1:
            GPU_STATUS = (GPU_STATUS & ~0x7FF) | (u32)(op & 0x7FF);
            Draw_SetTexpageMode((ParamDrawTexpageMode*)&op);
            break;
        case 0xE2:
            // https://psx-spx.consoledev.net/graphicsprocessingunitgpu/#gp0e2h-texture-window-setting
            Draw_SetTextureWindow(op & 0x1F, (op >> 5) & 0x1F,
                                  (op >> 10) & 0x1F, (op >> 15) & 0x1F);
            break;
        case 0xE3:
            Draw_SetAreaStart((int)op & 0x3FF, (int)(op >> 10) & 0x3FF);
            break;
        case 0xE4:
            Draw_SetAreaEnd((int)op & 0x3FF, (int)(op >> 10) & 0x3FF);
            break;
        case 0xE5:
            Draw_SetOffset((int)op & 0x7FF, (int)(op >> 11) & 0x7FF);
            break;
        case 0xE6:
            Draw_SetMask(!!(op & 1), !!(op & 2));
            break;
        default:
            if (code >= 0x20 && code < 0x80) {
                i += Draw_PushPrim(&buf[i], len - i) - 1;
                break;
            }
            WARNF("unsupported command %02X", code);
        }
    }
}

int Psyz_GpuExeque() {
    Draw_ResetBuffer();
    DispatchPackets(queue_buf, queue_len);
    Draw_FlushBuffer();
    Draw_ExequeSync();
    queue_len = 0;
    return queue_len;
}

static int GPU_Enqueue(u_long p1, u_long p2) {
    static unsigned long long chain_index;
    unsigned packet_index = 0;
    unsigned long long current_chain = chain_index++;
    int mask = (int)p2;
    if (mask) {
        WARNF("mask not supported (mask:%08X)", mask);
    }
    DR_ENV* env = (DR_ENV*)(uintptr_t)p1;
    while (1) {
        int env_len = getlen(env);
        TraceOtNode(current_chain, packet_index, env);
        if (env_len > 0x100) {
            ERRORF("packet %p is 0x%X words long, likely corrupted",
                   (void *)env, env_len);
            break;
        }
        if (queue_len + env_len > LEN(queue_buf)) {
            Psyz_GpuExeque();
        }
        {
            int canonical_len = CanonicalizePacket(
                env, queue_buf + queue_len, LEN(queue_buf) - queue_len);
            if (canonical_len < 0) {
                ERRORF("cannot canonicalize packet %p code=%02X len=%u",
                       (void*)env, getcode(env), getlen(env));
                break;
            }
            TraceCanonicalPacket(current_chain, packet_index++, getcode(env), env,
                                 queue_buf + queue_len, canonical_len);
        }
        queue_len += env_len;
        if (isendprim(env)) {
            break;
        }
        {
            DR_ENV* next = (DR_ENV*)nextPrim(env);
            if (((uintptr_t)next & (sizeof(u32) - 1)) != 0) {
                ERRORF("misaligned OT link %p after packet %p code=%02X len=%u",
                       (void *)next, (void *)env, getcode(env), getlen(env));
                break;
            }
            env = next;
        }
    }
    return 0;
}
static int GPU_DataWrite(u_long p1, u_long p2) {
    Psyz_GpuExeque();
    Draw_LoadImage((RECT*)(uintptr_t)p1, (u_long*)(uintptr_t)p2);
    return 0;
}
static int GPU_DataRead(u_long p1, u_long p2) {
    Psyz_GpuExeque();
    Draw_StoreImage((RECT*)(uintptr_t)p1, (u_long*)(uintptr_t)p2);
    return 0;
}

static int _param(int x) { return 0; }
static int psyz_addque2(
    int (*exec)(u_long p1, u_long p2), u_long p1, int len, u_long p2) {
    return exec(p1, p2);
}
static int psyz_addque(
    int (*exec)(u_long p1, u_long p2), u_long p1, u_long p2) {
    return psyz_addque2(exec, p1, 0, p2);
}

// psyz_clr is very similar to _clr from libgpu/sys.c
static DR_ENV clear_cmd;
static int psyz_clr(RECT* rect, u32 color) {
    const u32 wh = get_vram_wh();
    const unsigned short w = wh & 0xFFFF;
    const unsigned short h = wh >> 16;
    rect->w = CLAMP(rect->w, 0, w - 1);
    rect->h = CLAMP(rect->h, 0, h - 1);

#if 1 // HACK: faster but more inaccurate path, doesn't need a Psyz_GpuExeque
    setlen(&clear_cmd, 4);
    clear_cmd.code[0] = 0xE6000000; // mask bit setting
    clear_cmd.code[1] = (color & 0xFFFFFF) | 0x02000000;
    clear_cmd.code[2] = (u_long) * (u32*)&rect->x;
    clear_cmd.code[3] = (u_long) * (u32*)&rect->w;
#else
    // Needs a sync so 0xE1 CMDs can update GPU_STATUS before it gets used.
    Psyz_GpuExeque();
    setlen(&clear_cmd, 5);
    clear_cmd.code[0] = 0xE6000000; // mask bit setting
    clear_cmd.code[1] = 0xE1000000 | GPU_STATUS & 0x7FF | (color >> 0x1F) << 10;
    clear_cmd.code[2] = (color & 0xFFFFFF) | 0x02000000;
    clear_cmd.code[3] = (u_long) * (u32*)&rect->x;
    clear_cmd.code[4] = (u_long) * (u32*)&rect->w;
#endif

    termPrim(&clear_cmd);
    GPU_Enqueue((u_long)&clear_cmd, 0);
    return 0;
}

void Psyz_GpuDisplayCommand(unsigned int cmd) {
    unsigned char op = (cmd >> 24) & 0x3F;
    switch (op) {
    case 0:
        Draw_Reset();
        break;
    case 1:
        LOG_ONCE("Reset FIFO not implemented");
        break;
    case 2:
        LOG_ONCE("Ack IRQ not implemented");
        break;
    case 3:
        Draw_DisplayEnable(!(cmd & 1));
        break;
    case 4:
        LOG_ONCE("DMA direction not implemented");
        break;
    case 5:
        Draw_DisplayArea(cmd & 0x3FF, (cmd >> 10) & 0x3FF);
        break;
    case 6:
        Draw_DisplayHorizontalRange(cmd & 0xFFF, (cmd >> 12) & 0xFFF);
        break;
    case 7:
        Draw_DisplayVerticalRange(cmd & 0x3FF, (cmd >> 10) & 0x3FF);
        break;
    case 8:
        cmd &= 0xFFFFFF;
        Draw_SetDisplayMode((DisplayMode*)&cmd);
        break;
    default:
        WARNF("unhandled ctl %02X (%08X)", op, cmd);
        break;
    }
}
static int psyz_cwb() {
    NOT_IMPLEMENTED;
    return 0;
}
static int psyz_getctl(int _) {
    NOT_IMPLEMENTED;
    return 0;
}

static void psyz_otc(OT_TYPE* ot, s32 n) {
    s32 i;

    if (n <= 0) {
        return;
    }
    for (i = n - 1; i > 0; i--) {
        setaddr(&ot[i], &ot[i - 1]);
        setlen(&ot[i], 0);
    }
    setaddr(&ot[0], 0xFFFFFF);
    setlen(&ot[0], 0);
}

static int psyz_param(int _) {
    NOT_IMPLEMENTED;
    return 0;
}
static int psyz_reset(int _) { return GPU_V0; }
static u_long psyz_status(void) {
    NOT_IMPLEMENTED;
    return 0;
}
static int psyz_sync(int mode) {
    // see decomp/src/libgpu/sys.c
    // mode 0 waits until all the queue is drawn on screen
    // mode 1 process the queue and return how many elements have been queued
    // return -1 if GPU has timed out
    // but on PC the implementation is much simpler as it's always synced
    Psyz_GpuExeque();
    return 0;
}

int psyz_gpu_version(int mode) { return GPU_V0; }

// forwards raw GP0 words into the internal queue, then trigger execution.
void Psyz_GpuWriteGP0(unsigned int word) {
    if (queue_len >= (int)LEN(queue_buf)) {
        WARNF("GPU queue full");
        return;
    }
    queue_buf[queue_len++] = (u_long)word;
}

void GPU_cw(u_long* param) {
    struct Gpu* gpu = (struct Gpu*)param;
    gpu->ver = "psyz";
    gpu->addque = psyz_addque;
    gpu->addque2 = psyz_addque2;
    gpu->clr = psyz_clr;
    gpu->ctl = Psyz_GpuDisplayCommand;
    gpu->cwb = psyz_cwb;
    gpu->cwc = GPU_Enqueue;
    gpu->drs = GPU_DataRead;
    gpu->dws = GPU_DataWrite;
    gpu->exeque = Psyz_GpuExeque;
    gpu->getctl = psyz_getctl;
    gpu->otc = psyz_otc;
    gpu->param = psyz_param;
    gpu->reset = psyz_reset;
    gpu->status = psyz_status;
    gpu->sync = psyz_sync;
}

// these are not yet decompiled
int _addque2() { return 0; }
int _exeque() { return 0; }
int get_alarm(void) { return 0; }
