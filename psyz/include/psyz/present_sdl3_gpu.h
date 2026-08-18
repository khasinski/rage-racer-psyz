#ifndef PSYZ_PRESENT_SDL3_GPU_H
#define PSYZ_PRESENT_SDL3_GPU_H
#include <SDL3/SDL.h>

#ifdef __cplusplus
extern "C" {
#endif

// Source override consulted every presented frame, before the PS1 VRAM
// region is blitted to the swapchain. A host renderer (for example an
// enhanced/high-resolution scene renderer) fills `texture` to have its own
// image presented instead; leaving `texture` NULL keeps the normal PS1
// VRAM presentation.
typedef struct PsyzPresentSourceInfo {
    SDL_GPUTexture* texture; /* NULL: present PS1 VRAM as usual */
    Uint32 w, h;             /* source region, from (0,0) */
    float aspect;            /* presented aspect ratio; 0 means w/h */
    SDL_GPUFilter filter;    /* scaling filter, preset to NEAREST */
} PsyzPresentSourceInfo;

// Called with `info` zeroed apart from `filter`. The debug whole-VRAM view
// (F6) takes precedence over the override while active.
typedef void (*PsyzPresentSourceCB_SDL3GPU)(PsyzPresentSourceInfo* info);

// Register the present-source callback.
// Returns the previous callback, or NULL if none was set.
PsyzPresentSourceCB_SDL3GPU Psyz_PresentSource_SDL3GPU(
    PsyzPresentSourceCB_SDL3GPU cb);

// The native-resolution 1024x512 PS1 VRAM texture (RGBA8), for host
// renderers that sample game textures/CLUTs directly. NULL before platform
// initialization. Content reflects work submitted up to the previous
// present; sample it only from work submitted before the current one.
SDL_GPUTexture* Psyz_VideoGetVramTexture_SDL3GPU(void);

// Observe VRAM writes, in native PS1 pixel coordinates. A host renderer that
// caches decoded textures needs to know which regions stopped being valid,
// and cannot read the internal dirty rectangle without racing the reset that
// follows each sync.
//
// The callback runs inside the write path: keep it cheap and do not call back
// into the video API from it. Pass NULL to stop observing. Returns the
// previous observer.
typedef void (*PsyzVramWriteCB_SDL3GPU)(int x, int y, int w, int h);
PsyzVramWriteCB_SDL3GPU Psyz_VideoObserveVramWrites_SDL3GPU(
    PsyzVramWriteCB_SDL3GPU cb);

// Read back one region of VRAM as RGBA8888, w*h*4 bytes into out. Meant for
// decoding texture pages into host-side materials: a ray that hits a triangle
// cannot afford the CLUT indirection the rasterizer does per fragment.
//
// This stalls on the GPU, so it is for occasional use — texture pages are
// written when assets load and then stay put, which is what makes caching
// their decoded form worthwhile. Returns false when the region is out of
// bounds or the device is not up.
bool Psyz_VideoDownloadVramRegion_SDL3GPU(int x, int y, int w, int h,
                                          void* out);

#ifdef __cplusplus
}
#endif

#endif // PSYZ_PRESENT_SDL3_GPU_H
