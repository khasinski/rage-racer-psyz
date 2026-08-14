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

#ifdef __cplusplus
}
#endif

#endif // PSYZ_PRESENT_SDL3_GPU_H
