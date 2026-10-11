#include <psyz.h>
#include <psyz/overlay.h>
#include <psyz/overlay_sdl3_gpu.h>
#include <psyz/present_sdl3_gpu.h>
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "../internal.h"
#include <SDL3/SDL.h>
#include "sdl3_common.h"
#include "texture_sample.h"
#include "texture_span.h"
#include "texture_triangle.h"
#include "triangle_coverage.h"
#include "vram_read_cache.h"
#include "vram_sample_mask.h"

#include "shaders/psx_vert_spv.h"
#include "shaders/psx_frag_spv.h"
#include "shaders/clear_vert_spv.h"
#include "shaders/clear_frag_spv.h"
#include "shaders/psx_vert_msl.h"
#include "shaders/psx_frag_msl.h"
#include "shaders/clear_vert_msl.h"
#include "shaders/clear_frag_msl.h"
#include "shaders/present_vert_spv.h"
#include "shaders/present_frag_spv.h"
#include "shaders/present_vert_msl.h"
#include "shaders/present_frag_msl.h"

typedef struct {
    int x, y;
} Posi;

#define VRAM_BYTES (VRAM_W * VRAM_H * 4)

static SDL_GPUDevice* device = NULL;
static VramReadCache vram_read_cache;
static bool swapchain_ok = false;
static bool driver_vsync_requested;
static bool host_display_sync;
static SDL_GPUTexture* vram_render = NULL;
static VramSampleMask sample_dirty;
static bool external_vram_handle;
static SDL_GPUTexture* vram_sample = NULL;
static SDL_GPUTexture* auxiliary_vram = NULL;
static int texture_upload_context;
static SDL_GPUSampler* vram_sampler = NULL;
static SDL_GPUSampler* present_nearest_sampler = NULL;
static SDL_GPUSampler* present_linear_sampler = NULL;
static SDL_GPUTexture* scaled_vram_render = NULL;
static unsigned internal_res = 1;
static unsigned set_internal_res = 1;
static SDL_GPUBuffer* vbuf = NULL;
static SDL_GPUBuffer* ibuf = NULL;
static SDL_GPUBuffer* clear_vbuf = NULL;
static SDL_GPUTransferBuffer* vtx_transfer = NULL;
static SDL_GPUTransferBuffer* tex_upload_transfer = NULL;
static SDL_GPUTransferBuffer* tex_download_transfer = NULL;
static SDL_GPUGraphicsPipeline* pipe_tri_add = NULL;
static SDL_GPUGraphicsPipeline* pipe_tri_sub = NULL;
static SDL_GPUGraphicsPipeline* pipe_clear = NULL;
static SDL_GPUGraphicsPipeline* pipe_present = NULL;
static SDL_GPUCommandBuffer* pending_cmd = NULL;

static Posi display_area = {0, 0};
static Posi display_size = {256, 240};
static Posi cur_display_size = {-1, -1};
static Posi draw_offset = {0, 0};
static Posi draw_area_start = {0, 0};
static Posi draw_area_end = {0x10000, 0x10000};
static SDL_Rect scissor_rect = {0, 0, VRAM_W, VRAM_H};

static PsyzOverlayInitCB_SDL3GPU overlay_init_cb;
PsyzOverlayInitCB_SDL3GPU Psyz_OverlayInit_SDL3GPU(
    PsyzOverlayInitCB_SDL3GPU cb) {
    const PsyzOverlayInitCB_SDL3GPU prev = overlay_init_cb;
    overlay_init_cb = cb;
    return prev;
}

static PsyzOverlayRenderCB_SDL3GPU overlay_render_cb;
PsyzOverlayRenderCB_SDL3GPU Psyz_OverlayRender_SDL3GPU(
    PsyzOverlayRenderCB_SDL3GPU cb) {
    const PsyzOverlayRenderCB_SDL3GPU prev = overlay_render_cb;
    overlay_render_cb = cb;
    return prev;
}

static PsyzPresentSourceCB_SDL3GPU present_source_cb;
PsyzPresentSourceCB_SDL3GPU Psyz_PresentSource_SDL3GPU(
    PsyzPresentSourceCB_SDL3GPU cb) {
    const PsyzPresentSourceCB_SDL3GPU prev = present_source_cb;
    present_source_cb = cb;
    return prev;
}

static SDL_GPUCommandBuffer* AcquireCmd(void);
static void SubmitCmd(void);

SDL_GPUTexture* Psyz_VideoGetVramTexture_SDL3GPU(void) {
    /* Raw handle users can record writes outside this backend's observers. */
    if (vram_render) external_vram_handle = true;
    return vram_render;
}

bool Psyz_VideoGetPresentationDevice_SDL3GPU(SDL_Window** window,
                                          SDL_GPUDevice** gpu_device) {
    if (window) *window = NULL;
    if (gpu_device) *gpu_device = NULL;
    if (!window || !gpu_device || !is_platform_init_successful ||
        !device || !sdl3_window) return false;
    *window = sdl3_window;
    *gpu_device = device;
    return true;
}

SDL_GPUTexture* Psyz_VideoSnapshotVramTexture_SDL3GPU(void) {
    SDL_GPUCommandBuffer* cmd;
    SDL_GPUCopyPass* copy;
    const SDL_GPUTextureLocation source = {.texture = vram_render};
    const SDL_GPUTextureLocation destination = {.texture = vram_sample};

    if (!device || !vram_render || !vram_sample) {
        return NULL;
    }
    /* The modern renderer replays a captured packet list more than once
     * while interpolating a logic tick.  Do not leave those packets sampling
     * the live PS1 render target: later GP0 work may already be queued for it.
     * Freeze the complete texture pages and CLUTs in the backend's existing
     * sample mirror, then let every presentation of that captured frame read
     * the same image. */
    SubmitCmd();
    cmd = AcquireCmd();
    if (!cmd) return NULL;
    copy = SDL_BeginGPUCopyPass(cmd);
    if (!copy) return NULL;
    SDL_CopyGPUTextureToTexture(copy, &source, &destination,
                               VRAM_W, VRAM_H, 1, true);
    SDL_EndGPUCopyPass(copy);
    VramSampleMaskReset(&sample_dirty, false);
    SubmitCmd();
    return vram_sample;
}

static SDL_GPUCommandBuffer* AcquireCmd(void) {
    if (!pending_cmd) {
        pending_cmd = SDL_AcquireGPUCommandBuffer(device);
        if (!pending_cmd) {
            ERRORF("SDL_AcquireGPUCommandBuffer: %s", SDL_GetError());
        }
    }
    return pending_cmd;
}

static void SubmitCmd(void) {
    if (pending_cmd) {
        SDL_SubmitGPUCommandBuffer(pending_cmd);
        pending_cmd = NULL;
    }
}

static void SubmitCmdAndWait(void) {
    if (!pending_cmd) {
        return;
    }
    SDL_GPUFence* fence =
        SDL_SubmitGPUCommandBufferAndAcquireFence(pending_cmd);
    pending_cmd = NULL;
    if (fence) {
        SDL_WaitForGPUFences(device, true, &fence, 1);
        SDL_ReleaseGPUFence(device, fence);
    }
}

static SDL_GPUShader* CreateShader(
    const unsigned char* spirv, unsigned int spirv_len,
    const unsigned char* msl, unsigned int msl_len, SDL_GPUShaderStage stage,
    Uint32 num_samplers, Uint32 num_uniform_buffers) {
    const SDL_GPUShaderFormat formats = SDL_GetGPUShaderFormats(device);
    SDL_GPUShaderCreateInfo info = {
        .stage = stage,
        .num_samplers = num_samplers,
        .num_uniform_buffers = num_uniform_buffers,
    };
    if (formats & SDL_GPU_SHADERFORMAT_SPIRV) {
        info.code = spirv;
        info.code_size = spirv_len;
        info.format = SDL_GPU_SHADERFORMAT_SPIRV;
        info.entrypoint = "main";
    } else if (formats & SDL_GPU_SHADERFORMAT_MSL) {
        info.code = msl;
        info.code_size = msl_len;
        info.format = SDL_GPU_SHADERFORMAT_MSL;
        info.entrypoint = "main0";
    } else {
        ERRORF("no supported GPU shader format (formats=0x%x)", formats);
        return NULL;
    }
    SDL_GPUShader* shader = SDL_CreateGPUShader(device, &info);
    if (!shader) {
        ERRORF("SDL_CreateGPUShader: %s", SDL_GetError());
    }
    return shader;
}

static SDL_GPUGraphicsPipeline* CreatePsxPipeline(
    SDL_GPUShader* vs, SDL_GPUShader* fs, bool subtract) {
    const SDL_GPUVertexBufferDescription vb_desc = {
        .slot = 0,
        .pitch = sizeof(Vertex),
        .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
    };
    const SDL_GPUVertexAttribute attribs[] = {
        {.location = 0,
         .buffer_slot = 0,
         .format = SDL_GPU_VERTEXELEMENTFORMAT_SHORT2,
         .offset = offsetof(Vertex, x)},
        {.location = 1,
         .buffer_slot = 0,
         .format = SDL_GPU_VERTEXELEMENTFORMAT_USHORT4,
         .offset = offsetof(Vertex, u)},
        {.location = 2,
         .buffer_slot = 0,
         .format = SDL_GPU_VERTEXELEMENTFORMAT_UBYTE4_NORM,
         .offset = offsetof(Vertex, r)},
        {.location = 3,
         .buffer_slot = 0,
         .format = SDL_GPU_VERTEXELEMENTFORMAT_UBYTE4,
         .offset = offsetof(Vertex, twin)},
    };
    const SDL_GPUBlendOp op =
        subtract ? SDL_GPU_BLENDOP_REVERSE_SUBTRACT : SDL_GPU_BLENDOP_ADD;
    const SDL_GPUColorTargetDescription target = {
        .format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
        .blend_state =
            {
                .src_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE,
                .dst_color_blendfactor =
                    SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
                .color_blend_op = op,
                .src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE,
                .dst_alpha_blendfactor =
                    SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
                .alpha_blend_op = op,
                .enable_blend = true,
            },
    };
    const SDL_GPUGraphicsPipelineCreateInfo info = {
        .vertex_shader = vs,
        .fragment_shader = fs,
        .vertex_input_state =
            {
                .vertex_buffer_descriptions = &vb_desc,
                .num_vertex_buffers = 1,
                .vertex_attributes = attribs,
                .num_vertex_attributes = LENU(attribs),
            },
        .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
        .target_info =
            {
                .color_target_descriptions = &target,
                .num_color_targets = 1,
            },
    };
    SDL_GPUGraphicsPipeline* pipe =
        SDL_CreateGPUGraphicsPipeline(device, &info);
    if (!pipe) {
        ERRORF("SDL_CreateGPUGraphicsPipeline: %s", SDL_GetError());
    }
    return pipe;
}

// used only for a fast hardware ClearImage
static SDL_GPUGraphicsPipeline* CreateClearPipeline(
    SDL_GPUShader* vs, SDL_GPUShader* fs) {
    const SDL_GPUVertexBufferDescription vb_desc = {
        .slot = 0,
        .pitch = sizeof(Vertex),
        .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
    };
    const SDL_GPUVertexAttribute attribs[] = {
        {.location = 0,
         .buffer_slot = 0,
         .format = SDL_GPU_VERTEXELEMENTFORMAT_SHORT2,
         .offset = offsetof(Vertex, x)},
        {.location = 1,
         .buffer_slot = 0,
         .format = SDL_GPU_VERTEXELEMENTFORMAT_UBYTE4_NORM,
         .offset = offsetof(Vertex, r)},
    };
    // blending disabled: the clear color (including alpha 0, the PS1 mask
    // bit) is written as-is, matching glClearColor(r, g, b, 0) + glClear
    const SDL_GPUColorTargetDescription target = {
        .format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
    };
    const SDL_GPUGraphicsPipelineCreateInfo info = {
        .vertex_shader = vs,
        .fragment_shader = fs,
        .vertex_input_state =
            {
                .vertex_buffer_descriptions = &vb_desc,
                .num_vertex_buffers = 1,
                .vertex_attributes = attribs,
                .num_vertex_attributes = LENU(attribs),
            },
        .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLESTRIP,
        .target_info =
            {
                .color_target_descriptions = &target,
                .num_color_targets = 1,
            },
    };
    SDL_GPUGraphicsPipeline* pipe =
        SDL_CreateGPUGraphicsPipeline(device, &info);
    if (!pipe) {
        ERRORF("SDL_CreateGPUGraphicsPipeline: %s", SDL_GetError());
    }
    return pipe;
}

static SDL_GPUGraphicsPipeline* CreatePresentPipeline(
    SDL_GPUShader* vs, SDL_GPUShader* fs) {
    const SDL_GPUColorTargetDescription target = {
        .format = SDL_GetGPUSwapchainTextureFormat(device, sdl3_window),
    };
    const SDL_GPUGraphicsPipelineCreateInfo info = {
        .vertex_shader = vs,
        .fragment_shader = fs,
        .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
        .target_info = {
            .color_target_descriptions = &target,
            .num_color_targets = 1,
        },
    };
    SDL_GPUGraphicsPipeline* pipe =
        SDL_CreateGPUGraphicsPipeline(device, &info);
    if (!pipe) {
        ERRORF("SDL_CreateGPUGraphicsPipeline (present): %s", SDL_GetError());
    }
    return pipe;
}

static bool CreateGpuResources(void) {
    VramSampleMaskReset(&sample_dirty, true);
    external_vram_handle = false;
    const SDL_GPUTextureCreateInfo render_info = {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
        .usage =
            SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER,
        .width = VRAM_W,
        .height = VRAM_H,
        .layer_count_or_depth = 1,
        .num_levels = 1,
    };
    vram_render = SDL_CreateGPUTexture(device, &render_info);
    SDL_GPUTextureCreateInfo sample_info = render_info;
    sample_info.usage = SDL_GPU_TEXTUREUSAGE_SAMPLER;
    vram_sample = SDL_CreateGPUTexture(device, &sample_info);
    auxiliary_vram = SDL_CreateGPUTexture(device, &sample_info);
    if (!vram_render || !vram_sample || !auxiliary_vram) {
        ERRORF("SDL_CreateGPUTexture: %s", SDL_GetError());
        return false;
    }

    const SDL_GPUSamplerCreateInfo sampler_info = {
        .min_filter = SDL_GPU_FILTER_NEAREST,
        .mag_filter = SDL_GPU_FILTER_NEAREST,
        .mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST,
        .address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        .address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        .address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
    };
    vram_sampler = SDL_CreateGPUSampler(device, &sampler_info);
    present_nearest_sampler = SDL_CreateGPUSampler(device, &sampler_info);
    SDL_GPUSamplerCreateInfo linear_sampler_info = sampler_info;
    linear_sampler_info.min_filter = SDL_GPU_FILTER_LINEAR;
    linear_sampler_info.mag_filter = SDL_GPU_FILTER_LINEAR;
    present_linear_sampler =
        SDL_CreateGPUSampler(device, &linear_sampler_info);
    if (!vram_sampler || !present_nearest_sampler || !present_linear_sampler) {
        ERRORF("SDL_CreateGPUSampler: %s", SDL_GetError());
        return false;
    }

    const SDL_GPUBufferCreateInfo vbuf_info = {
        .usage = SDL_GPU_BUFFERUSAGE_VERTEX,
        .size = MAX_VERTEX_COUNT * sizeof(Vertex),
    };
    vbuf = SDL_CreateGPUBuffer(device, &vbuf_info);
    const SDL_GPUBufferCreateInfo ibuf_info = {
        .usage = SDL_GPU_BUFFERUSAGE_INDEX,
        .size = MAX_INDEX_COUNT * sizeof(u16),
    };
    ibuf = SDL_CreateGPUBuffer(device, &ibuf_info);
    const SDL_GPUBufferCreateInfo clear_vbuf_info = {
        .usage = SDL_GPU_BUFFERUSAGE_VERTEX,
        .size = 4 * sizeof(Vertex),
    };
    clear_vbuf = SDL_CreateGPUBuffer(device, &clear_vbuf_info);
    if (!vbuf || !ibuf || !clear_vbuf) {
        ERRORF("SDL_CreateGPUBuffer: %s", SDL_GetError());
        return false;
    }

    const SDL_GPUTransferBufferCreateInfo vtx_transfer_info = {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size =
            MAX_VERTEX_COUNT * sizeof(Vertex) + MAX_INDEX_COUNT * sizeof(u16),
    };
    vtx_transfer = SDL_CreateGPUTransferBuffer(device, &vtx_transfer_info);
    const SDL_GPUTransferBufferCreateInfo upload_info = {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = VRAM_BYTES,
    };
    tex_upload_transfer = SDL_CreateGPUTransferBuffer(device, &upload_info);
    const SDL_GPUTransferBufferCreateInfo download_info = {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD,
        .size = VRAM_BYTES,
    };
    tex_download_transfer = SDL_CreateGPUTransferBuffer(device, &download_info);
    if (!vtx_transfer || !tex_upload_transfer || !tex_download_transfer) {
        ERRORF("SDL_CreateGPUTransferBuffer: %s", SDL_GetError());
        return false;
    }

    /* Real PS1 VRAM is zero after GPU reset. SDL GPU texture contents are
     * undefined at creation, and Rage legitimately samples untouched texels
     * around packed 4-bit assets. Keep both the render and sampling copies in
     * the same deterministic reset state before the first upload. */
    u8* initial_vram = SDL_MapGPUTransferBuffer(device, tex_upload_transfer, true);
    if (!initial_vram) {
        ERRORF("SDL_MapGPUTransferBuffer: %s", SDL_GetError());
        return false;
    }
    memset(initial_vram, 0, VRAM_BYTES);
    SDL_UnmapGPUTransferBuffer(device, tex_upload_transfer);
    SDL_GPUCommandBuffer* init_cmd = SDL_AcquireGPUCommandBuffer(device);
    if (!init_cmd) {
        ERRORF("SDL_AcquireGPUCommandBuffer: %s", SDL_GetError());
        return false;
    }
    SDL_GPUCopyPass* init_copy = SDL_BeginGPUCopyPass(init_cmd);
    const SDL_GPUTextureTransferInfo init_src = {
        .transfer_buffer = tex_upload_transfer,
        .pixels_per_row = VRAM_W,
        .rows_per_layer = VRAM_H,
    };
    const SDL_GPUTextureRegion init_render = {
        .texture = vram_render, .w = VRAM_W, .h = VRAM_H, .d = 1};
    const SDL_GPUTextureRegion init_sample = {
        .texture = vram_sample, .w = VRAM_W, .h = VRAM_H, .d = 1};
    SDL_UploadToGPUTexture(init_copy, &init_src, &init_render, false);
    SDL_UploadToGPUTexture(init_copy, &init_src, &init_sample, false);
    SDL_EndGPUCopyPass(init_copy);
    SDL_GPUFence* init_fence = SDL_SubmitGPUCommandBufferAndAcquireFence(init_cmd);
    if (!init_fence) {
        ERRORF("SDL_SubmitGPUCommandBufferAndAcquireFence: %s", SDL_GetError());
        return false;
    }
    SDL_WaitForGPUFences(device, true, &init_fence, 1);
    SDL_ReleaseGPUFence(device, init_fence);

    SDL_GPUShader* psx_vs =
        CreateShader(psx_vert_spv, psx_vert_spv_len, psx_vert_msl,
                     psx_vert_msl_len, SDL_GPU_SHADERSTAGE_VERTEX, 0, 1);
    SDL_GPUShader* psx_fs =
        CreateShader(psx_frag_spv, psx_frag_spv_len, psx_frag_msl,
                     psx_frag_msl_len, SDL_GPU_SHADERSTAGE_FRAGMENT, 1, 0);
    SDL_GPUShader* clear_vs =
        CreateShader(clear_vert_spv, clear_vert_spv_len, clear_vert_msl,
                     clear_vert_msl_len, SDL_GPU_SHADERSTAGE_VERTEX, 0, 0);
    SDL_GPUShader* clear_fs =
        CreateShader(clear_frag_spv, clear_frag_spv_len, clear_frag_msl,
                     clear_frag_msl_len, SDL_GPU_SHADERSTAGE_FRAGMENT, 0, 0);
    SDL_GPUShader* present_vs =
        CreateShader(present_vert_spv, present_vert_spv_len, present_vert_msl,
                     present_vert_msl_len, SDL_GPU_SHADERSTAGE_VERTEX, 0, 0);
    SDL_GPUShader* present_fs =
        CreateShader(present_frag_spv, present_frag_spv_len, present_frag_msl,
                     present_frag_msl_len, SDL_GPU_SHADERSTAGE_FRAGMENT, 1, 1);
    bool shaders_ok = psx_vs && psx_fs && clear_vs && clear_fs &&
                      present_vs && present_fs;
    if (shaders_ok) {
        pipe_tri_add = CreatePsxPipeline(psx_vs, psx_fs, false);
        pipe_tri_sub = CreatePsxPipeline(psx_vs, psx_fs, true);
        pipe_clear = CreateClearPipeline(clear_vs, clear_fs);
        if (swapchain_ok) {
            pipe_present = CreatePresentPipeline(present_vs, present_fs);
        }
    }
    if (psx_vs) {
        SDL_ReleaseGPUShader(device, psx_vs);
    }
    if (psx_fs) {
        SDL_ReleaseGPUShader(device, psx_fs);
    }
    if (clear_vs) {
        SDL_ReleaseGPUShader(device, clear_vs);
    }
    if (clear_fs) {
        SDL_ReleaseGPUShader(device, clear_fs);
    }
    if (present_vs) {
        SDL_ReleaseGPUShader(device, present_vs);
    }
    if (present_fs) {
        SDL_ReleaseGPUShader(device, present_fs);
    }
    return shaders_ok && pipe_tri_add && pipe_tri_sub && pipe_clear &&
           (!swapchain_ok || pipe_present);
}

bool InitPlatform() {
    if (is_platform_initialized) {
        return is_platform_init_successful;
    }
    is_platform_initialized = true; // don't re-initialize on failures

    ConfigureTestWindow();
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        ERRORF("SDL_Init: %s", SDL_GetError());
        return false;
    }

    device = SDL_CreateGPUDevice(
        SDL_GPU_SHADERFORMAT_SPIRV | SDL_GPU_SHADERFORMAT_MSL, false, NULL);
    if (!device) {
        ERRORF("SDL_CreateGPUDevice: %s", SDL_GetError());
        return false;
    }
    INFOF("SDL_GPU with driver %s", SDL_GetGPUDeviceDriver(device));

    sdl3_window = SDL_CreateWindow(
        window_title, DEFAULT_FRONT_W, DEFAULT_FRONT_H,
        SDL_WINDOW_HIDDEN | SDL_WINDOW_RESIZABLE |
            SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!sdl3_window) {
        ERRORF("SDL_CreateWindow: %s", SDL_GetError());
        return false;
    }

    /* Complete the X11 map handshake before Vulkan WSI starts consuming
     * window events. Showing the hidden window after claiming its swapchain
     * can leave X11_ShowWindow waiting forever for a MapNotify. */
    if (!headless_window && !SDL_ShowWindow(sdl3_window)) {
        ERRORF("SDL_ShowWindow: %s", SDL_GetError());
        return false;
    }
    is_window_visible = !headless_window;

    // useful for offscreen rendering with SDL_VIDEODRIVER=offscreen
    swapchain_ok = !headless_window && SDL_ClaimWindowForGPUDevice(device, sdl3_window);
    if (!swapchain_ok && !headless_window) {
        WARNF("no swapchain, presentation disabled: %s", SDL_GetError());
    } else if (swapchain_ok && !SDL_SetGPUAllowedFramesInFlight(device, 1)) {
        WARNF("could not select low-latency presentation: %s", SDL_GetError());
    }

    if (!CreateGpuResources()) {
        return false;
    }

    cur_tpage = 0;
    Sdl3Common_TimingInit();

    is_platform_init_successful = true;
    if (overlay_init_cb) {
        overlay_init_cb(sdl3_window, device);
    }
    return true;
}

static void PlatformBackend_SetDriverVsync(bool enable) {
    driver_vsync_requested = enable;
    if (!device || !swapchain_ok) {
        return;
    }
    SDL_GPUPresentMode mode = SDL_GPU_PRESENTMODE_VSYNC;
    if (!enable && !host_display_sync) {
        if (SDL_WindowSupportsGPUPresentMode(
                device, sdl3_window, SDL_GPU_PRESENTMODE_IMMEDIATE)) {
            mode = SDL_GPU_PRESENTMODE_IMMEDIATE;
        } else if (SDL_WindowSupportsGPUPresentMode(
                       device, sdl3_window, SDL_GPU_PRESENTMODE_MAILBOX)) {
            mode = SDL_GPU_PRESENTMODE_MAILBOX;
        }
    }
    if (!SDL_SetGPUSwapchainParameters(
            device, sdl3_window, SDL_GPU_SWAPCHAINCOMPOSITION_SDR, mode)) {
        WARNF("could not set presentation synchronization: %s", SDL_GetError());
    } else {
        INFOF("presentation mode=%s host_display_sync=%d logic_driver_vsync=%d",
            mode == SDL_GPU_PRESENTMODE_VSYNC ? "vsync" :
            mode == SDL_GPU_PRESENTMODE_MAILBOX ? "mailbox" : "immediate",
            (int)host_display_sync, (int)driver_vsync_requested);
    }
}

static void UpdateScissor(void);

static SDL_GPUTexture* GetRenderTarget(void) {
    return internal_res <= 1 ? vram_render : scaled_vram_render;
}

static void MarkVramDirty(SDL_Rect r) {
    if (r.w <= 0 || r.h <= 0) {
        return;
    }
    int x0 = CLAMP(r.x, 0, VRAM_W);
    int y0 = CLAMP(r.y, 0, VRAM_H);
    int x1 = CLAMP(r.x + r.w, 0, VRAM_W);
    int y1 = CLAMP(r.y + r.h, 0, VRAM_H);
    if (x1 <= x0 || y1 <= y0) {
        return;
    }
    VramReadCacheInvalidate(&vram_read_cache, x0, y0, x1 - x0, y1 - y0);
    VramSampleMaskMark(&sample_dirty, x0, y0, x1 - x0, y1 - y0);
}

// mirror a native VRAM region into the scaled render target
static void SyncNativeVramToScaled(int x, int y, int w, int h) {
    if (internal_res <= 1 || !scaled_vram_render || w <= 0 || h <= 0) {
        return;
    }
    SDL_GPUCommandBuffer* cmd = AcquireCmd();
    if (!cmd) {
        return;
    }
    const SDL_GPUBlitInfo blit = {
        .source = {.texture = vram_render,
                   .x = (Uint32)x,
                   .y = (Uint32)y,
                   .w = (Uint32)w,
                   .h = (Uint32)h},
        .destination = {.texture = scaled_vram_render,
                        .x = (Uint32)(x * internal_res),
                        .y = (Uint32)(y * internal_res),
                        .w = (Uint32)(w * internal_res),
                        .h = (Uint32)(h * internal_res)},
        .load_op = SDL_GPU_LOADOP_LOAD,
        .filter = SDL_GPU_FILTER_NEAREST,
    };
    SDL_BlitGPUTexture(cmd, &blit);
}

static void SyncScaledVramToNative(void) {
    if (internal_res <= 1 || !scaled_vram_render) {
        return;
    }
    int x0 = CLAMP(draw_area_start.x, 0, VRAM_W);
    int y0 = CLAMP(draw_area_start.y, 0, VRAM_H);
    int x1 = CLAMP(draw_area_end.x + 1, 0, VRAM_W);
    int y1 = CLAMP(draw_area_end.y + 1, 0, VRAM_H);
    if (x1 <= x0 || y1 <= y0) {
        return;
    }
    SDL_GPUCommandBuffer* cmd = AcquireCmd();
    if (!cmd) {
        return;
    }
    const SDL_GPUBlitInfo blit = {
        .source = {.texture = scaled_vram_render,
                   .x = (Uint32)(x0 * internal_res),
                   .y = (Uint32)(y0 * internal_res),
                   .w = (Uint32)((x1 - x0) * internal_res),
                   .h = (Uint32)((y1 - y0) * internal_res)},
        .destination = {.texture = vram_render,
                        .x = (Uint32)x0,
                        .y = (Uint32)y0,
                        .w = (Uint32)(x1 - x0),
                        .h = (Uint32)(y1 - y0)},
        .load_op = SDL_GPU_LOADOP_LOAD,
        .filter = SDL_GPU_FILTER_NEAREST,
    };
    SDL_BlitGPUTexture(cmd, &blit);
    MarkVramDirty((SDL_Rect){x0, y0, x1 - x0, y1 - y0});
}

static bool CreateScaledVramTarget(unsigned n) {
    if (scaled_vram_render) {
        SDL_ReleaseGPUTexture(device, scaled_vram_render);
        scaled_vram_render = NULL;
    }
    const SDL_GPUTextureCreateInfo info = {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
        .usage =
            SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER,
        .width = VRAM_W * n,
        .height = VRAM_H * n,
        .layer_count_or_depth = 1,
        .num_levels = 1,
    };
    scaled_vram_render = SDL_CreateGPUTexture(device, &info);
    if (!scaled_vram_render) {
        ERRORF(
            "scaled VRAM texture creation failed (%dx): %s", n, SDL_GetError());
        return false;
    }
    return true;
}

// ensure scaled VRAM does not exceed max allowed texture size
static unsigned AdjustInternalRes(unsigned scale) {
    unsigned n = scale;
    const unsigned max_dim = 16384; // max theoretical texture size
    unsigned vram_dim = VRAM_W > VRAM_H ? VRAM_W : VRAM_H;
    while (n > 1 && vram_dim * n > max_dim) {
        n--;
    }
    if (n != scale) {
        WARNF("internal resolution %dx exceeds max texture size, "
              "fallback to internal resolution %dx",
              scale, n);
    }
    return n;
}

static void ApplyPendingInternalRes(void) {
    if (set_internal_res == internal_res) {
        return;
    }
    if (!sdl3_window || !is_platform_init_successful) {
        return;
    }
    unsigned n = AdjustInternalRes(set_internal_res);
    if (n == internal_res) {
        return;
    }
    Draw_FlushBuffer(); // render pending prims with the old multiplier
    if (n > 1 && !CreateScaledVramTarget(n)) {
        n = 1;
        set_internal_res = 1;
    }
    if (n > 1) {
        // upscale the whole native VRAM into the freshly created scaled target
        SDL_GPUCommandBuffer* cmd = AcquireCmd();
        if (cmd) {
            const SDL_GPUBlitInfo blit = {
                .source = {.texture = vram_render, .w = VRAM_W, .h = VRAM_H},
                .destination = {.texture = scaled_vram_render,
                                .w = VRAM_W * n,
                                .h = VRAM_H * n},
                .load_op = SDL_GPU_LOADOP_DONT_CARE,
                .filter = SDL_GPU_FILTER_NEAREST,
            };
            SDL_BlitGPUTexture(cmd, &blit);
        }
    } else if (scaled_vram_render) {
        SDL_ReleaseGPUTexture(device, scaled_vram_render);
        scaled_vram_render = NULL;
    }
    internal_res = n;
    UpdateScissor();
    INFOF("internal resolution set to %dx (%dx%d)", n, VRAM_W * n, VRAM_H * n);
}

static void PlatformBackend_Present(void) {
    PsyzPresentSourceInfo present_source = {0};
    bool queued_present = false;
    if (!sdl3_window && !InitPlatform()) {
        return;
    }

    ApplyPendingInternalRes();

    if (present_source_cb) {
        // Flush accumulated game work (draws, VRAM moves such as texture
        // animation) before the host renderer samples the VRAM texture, so
        // it never observes a half-applied transfer.
        SubmitCmd();
        /* The source callback may submit GPU work of its own.  Do that before
         * acquiring the command buffer used to draw the swapchain: SDL's
         * Vulkan backend recycles command-buffer descriptor caches on submit,
         * so nesting a submission while this buffer is open can leave the
         * present draw with a stale cache. */
        present_source.filter = SDL_GPU_FILTER_NEAREST;
        present_source_cb(&present_source);
        if (host_display_sync != present_source.sync_to_display) {
            host_display_sync = present_source.sync_to_display;
            /* This changes swapchain scheduling only. The common emulated
             * VBlank clock/use_driver_vsync value remains untouched. */
            PlatformBackend_SetDriverVsync(driver_vsync_requested);
        }
        /* A host may pace presentation independently of emulated VBlank.
         * Do not queue a duplicate image at a logic-only tick. */
        if (present_source.skip_present && !debug_show_vram) return;
    } else if (host_display_sync) {
        host_display_sync = false;
        PlatformBackend_SetDriverVsync(driver_vsync_requested);
    }

    SDL_GPUCommandBuffer* cmd = AcquireCmd();
    if (!cmd) {
        return;
    }
    if (swapchain_ok) {
        SDL_GPUTexture* swapchain = NULL;
        Uint32 sc_w = 0, sc_h = 0;
        if (present_nonblocking_request) {
            // Skip the frame when no swapchain image is free instead of
            // stalling the caller (the game's frame-sync wait).
            if (!SDL_AcquireGPUSwapchainTexture(
                    cmd, sdl3_window, &swapchain, &sc_w, &sc_h)) {
                swapchain = NULL;
            }
        } else if (!SDL_WaitAndAcquireGPUSwapchainTexture(
                       cmd, sdl3_window, &swapchain, &sc_w, &sc_h)) {
            WARNF("SDL_WaitAndAcquireGPUSwapchainTexture: %s", SDL_GetError());
        }
        if (swapchain) {
            queued_present = true;
            const Uint32 n = internal_res;
            SDL_GPUBlitRegion src = {
                .texture = GetRenderTarget(),
                .x = (Uint32)display_area.x * n,
                .y = (Uint32)display_area.y * n,
                .w = (Uint32)display_size.x * n,
                .h = (Uint32)display_size.y * n,
            };
            Uint32 source_w = VRAM_W * n;
            Uint32 source_h = VRAM_H * n;
            float game_aspect =
                GetCurrentGameAspectRatio(display_size.x, display_size.y);
            SDL_GPUFilter present_filter = SDL_GPU_FILTER_NEAREST;
            if (debug_show_vram) {
                src.x = 0;
                src.y = 0;
                src.w = VRAM_W * n;
                src.h = VRAM_H * n;
                game_aspect = (float)VRAM_W / (float)VRAM_H;
            } else if (present_source_cb) {
                if (present_source.texture && present_source.w > 0 &&
                    present_source.h > 0) {
                    src.texture = present_source.texture;
                    src.x = 0;
                    src.y = 0;
                    src.w = present_source.w;
                    src.h = present_source.h;
                    source_w = present_source.w;
                    source_h = present_source.h;
                    game_aspect = present_source.aspect > 0.0f
                                      ? present_source.aspect
                                      : (float)present_source.w /
                                            (float)present_source.h;
                    present_filter = present_source.filter;
                }
            }

            WndSize win = {(int)sc_w, (int)sc_h};
            SDL_Rect dst = FitGameToWindow(game_aspect, win);

            const SDL_GPUColorTargetInfo target = {
                .texture = swapchain,
                .clear_color = {0.0f, 0.0f, 0.0f, 1.0f},
                .load_op = SDL_GPU_LOADOP_CLEAR,
                .store_op = SDL_GPU_STOREOP_STORE,
            };
            SDL_GPURenderPass* pass =
                SDL_BeginGPURenderPass(cmd, &target, 1, NULL);
            if (disp_on) {
                const SDL_GPUViewport viewport = {
                    .x = (float)dst.x,
                    .y = (float)dst.y,
                    .w = (float)dst.w,
                    .h = (float)dst.h,
                    .min_depth = 0.0f,
                    .max_depth = 1.0f,
                };
                const float uv_rect[4] = {
                    (float)src.x / (float)source_w,
                    (float)src.y / (float)source_h,
                    (float)src.w / (float)source_w,
                    (float)src.h / (float)source_h,
                };
                const SDL_GPUTextureSamplerBinding binding = {
                    .texture = src.texture,
                    .sampler = present_filter == SDL_GPU_FILTER_LINEAR
                                   ? present_linear_sampler
                                   : present_nearest_sampler,
                };
                SDL_BindGPUGraphicsPipeline(pass, pipe_present);
                SDL_SetGPUViewport(pass, &viewport);
                SDL_BindGPUFragmentSamplers(pass, 0, &binding, 1);
                SDL_PushGPUFragmentUniformData(
                    cmd, 0, uv_rect, sizeof(uv_rect));
                SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
            }
            SDL_EndGPURenderPass(pass);
            if (overlay_frame_cb) {
                overlay_frame_cb();
            }
            if (overlay_render_cb) {
                SDL_GPUColorTargetInfo target = {
                    .texture = swapchain,
                    .load_op = SDL_GPU_LOADOP_LOAD,
                    .store_op = SDL_GPU_STOREOP_STORE,
                };
                SDL_GPURenderPass* pass =
                    SDL_BeginGPURenderPass(cmd, &target, 1, NULL);
                overlay_render_cb(cmd, pass);
                SDL_EndGPURenderPass(pass);
            }
        }
    } else if (overlay_frame_cb) {
        overlay_frame_cb();
    }
    finish_time = SDL_GetPerformanceCounter();
    SubmitCmd();
    if (queued_present) gpu_stats.presented_frames++;
}

static void QuitPlatform(void) {
    host_display_sync = false;
    VramReadCacheReset(&vram_read_cache);
    /* Retire encoded work before clients release resources referenced by it.
     * Waiting alone does not submit the backend's pending command buffer. */
    if (device) {
        SubmitCmd();
        SDL_WaitForGPUIdle(device);
    }
    /* Failed/never-started devices have no published presentation handles.
     * Clients must not receive a destroy event for an unpublished lifetime. */
    if (is_platform_init_successful && overlay_destroy_cb) {
        overlay_destroy_cb();
    }
    if (device) {
        if (pipe_tri_add) {
            SDL_ReleaseGPUGraphicsPipeline(device, pipe_tri_add);
            pipe_tri_add = NULL;
        }
        if (pipe_tri_sub) {
            SDL_ReleaseGPUGraphicsPipeline(device, pipe_tri_sub);
            pipe_tri_sub = NULL;
        }
        if (pipe_clear) {
            SDL_ReleaseGPUGraphicsPipeline(device, pipe_clear);
            pipe_clear = NULL;
        }
        if (pipe_present) {
            SDL_ReleaseGPUGraphicsPipeline(device, pipe_present);
            pipe_present = NULL;
        }
        if (vram_render) {
            SDL_ReleaseGPUTexture(device, vram_render);
            vram_render = NULL;
        }
        if (vram_sample) {
            SDL_ReleaseGPUTexture(device, vram_sample);
            vram_sample = NULL;
        }
        if (auxiliary_vram) {
            SDL_ReleaseGPUTexture(device, auxiliary_vram);
            auxiliary_vram = NULL;
        }
        if (scaled_vram_render) {
            SDL_ReleaseGPUTexture(device, scaled_vram_render);
            scaled_vram_render = NULL;
        }
        if (vram_sampler) {
            SDL_ReleaseGPUSampler(device, vram_sampler);
            vram_sampler = NULL;
        }
        if (present_nearest_sampler) {
            SDL_ReleaseGPUSampler(device, present_nearest_sampler);
            present_nearest_sampler = NULL;
        }
        if (present_linear_sampler) {
            SDL_ReleaseGPUSampler(device, present_linear_sampler);
            present_linear_sampler = NULL;
        }
        if (vbuf) {
            SDL_ReleaseGPUBuffer(device, vbuf);
            vbuf = NULL;
        }
        if (ibuf) {
            SDL_ReleaseGPUBuffer(device, ibuf);
            ibuf = NULL;
        }
        if (clear_vbuf) {
            SDL_ReleaseGPUBuffer(device, clear_vbuf);
            clear_vbuf = NULL;
        }
        if (vtx_transfer) {
            SDL_ReleaseGPUTransferBuffer(device, vtx_transfer);
            vtx_transfer = NULL;
        }
        if (tex_upload_transfer) {
            SDL_ReleaseGPUTransferBuffer(device, tex_upload_transfer);
            tex_upload_transfer = NULL;
        }
        if (tex_download_transfer) {
            SDL_ReleaseGPUTransferBuffer(device, tex_download_transfer);
            tex_download_transfer = NULL;
        }
        if (swapchain_ok) {
            SDL_ReleaseWindowFromGPUDevice(device, sdl3_window);
            swapchain_ok = false;
        }
        SDL_DestroyGPUDevice(device);
        device = NULL;
    }
    if (sdl3_window) {
        SDL_DestroyWindow(sdl3_window);
        sdl3_window = NULL;
        is_window_visible = false;
    }
    /* SDL_Quit destroys device streams too; clear our owner first so a later
     * platform/audio initialization cannot reuse dangling stream state. */
    Psyz_AudioDestroy();
    SDL_Quit();
    is_platform_initialized = false;
    is_platform_init_successful = false;
}

void ResetPlatform(void) {
    cur_tpage = 0;
    internal_res = 1;
    QuitPlatform();
}

static bool DownloadVramRegionAsRGBA8888(int x, int y, int w, int h, u8* out) {
    SDL_GPUCommandBuffer* cmd = AcquireCmd();
    if (!cmd) {
        return false;
    }
    SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
    const SDL_GPUTextureRegion region = {
        .texture = vram_render,
        .x = (Uint32)x,
        .y = (Uint32)y,
        .w = (Uint32)w,
        .h = (Uint32)h,
        .d = 1,
    };
    const SDL_GPUTextureTransferInfo transfer = {
        .transfer_buffer = tex_download_transfer,
    };
    SDL_DownloadFromGPUTexture(copy, &region, &transfer);
    SDL_EndGPUCopyPass(copy);
    SubmitCmdAndWait();

    const u8* map =
        SDL_MapGPUTransferBuffer(device, tex_download_transfer, false);
    if (!map) {
        ERRORF("SDL_MapGPUTransferBuffer: %s", SDL_GetError());
        return false;
    }
    memcpy(out, map, (size_t)w * h * 4);
    SDL_UnmapGPUTransferBuffer(device, tex_download_transfer);
    return true;
}

unsigned char* Psyz_VideoAllocCapturedFrame(int* w, int* h) {
    const int channels = 3;
    if (!device || !vram_render) {
        *w = *h = 0;
        ERRORF("GPU device not initialized");
        return NULL;
    }

    *w = display_size.x;
    *h = display_size.y;
    unsigned char* pixels = malloc((*w) * (*h) * channels);
    u8* rgba = malloc((size_t)(*w) * (*h) * 4);
    if (!pixels || !rgba) {
        free(pixels);
        free(rgba);
        return NULL;
    }

    if (!DownloadVramRegionAsRGBA8888(
            display_area.x, display_area.y, *w, *h, rgba)) {
        free(pixels);
        free(rgba);
        return NULL;
    }
    const u8* src = rgba;
    unsigned char* dst = pixels;
    for (int i = 0; i < (*w) * (*h); i++) {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst += 3;
        src += 4;
    }
    free(rgba);
    return pixels;
}

unsigned char* Psyz_VideoAllocCapturedDrawPage(int* w, int* h) {
    const int channels = 3;
    int page_y;
    unsigned char* pixels;
    u8* rgba;
    if (!device || !vram_render) {
        *w = *h = 0;
        return NULL;
    }
    *w = display_size.x;
    *h = display_size.y;
    page_y = (draw_area_start.y / 240) * 240;
    pixels = malloc((size_t)(*w) * (*h) * channels);
    rgba = malloc((size_t)(*w) * (*h) * 4);
    if (!pixels || !rgba) {
        free(pixels);
        free(rgba);
        return NULL;
    }
    if (!DownloadVramRegionAsRGBA8888(0, page_y, *w, *h, rgba)) {
        free(pixels);
        free(rgba);
        return NULL;
    }
    for (int i = 0; i < (*w) * (*h); i++) {
        pixels[i * 3 + 0] = rgba[i * 4 + 0];
        pixels[i * 3 + 1] = rgba[i * 4 + 1];
        pixels[i * 3 + 2] = rgba[i * 4 + 2];
    }
    free(rgba);
    return pixels;
}

unsigned short* Psyz_VideoAllocCapturedVram(int* w, int* h) {
    size_t pixels;
    u8* rgba;
    unsigned short* output;
    if (!device || !vram_render) {
        *w = *h = 0;
        return NULL;
    }
    Draw_FlushBuffer();
    *w = VRAM_W;
    *h = VRAM_H;
    pixels = (size_t)*w * *h;
    rgba = malloc(pixels * 4);
    output = malloc(pixels * sizeof(*output));
    if (!rgba || !output ||
        !DownloadVramRegionAsRGBA8888(0, 0, *w, *h, rgba)) {
        free(rgba);
        free(output);
        return NULL;
    }
    ConvertRgba8888ToRgb5551(rgba, output, pixels);
    free(rgba);
    return output;
}

int Psyz_VideoGetCaptureState(PsyzVideoCaptureState* state) {
    if (!state) return -1;
    state->draw_x = draw_area_start.x;
    state->draw_y = draw_area_start.y;
    state->draw_w = draw_area_end.x - draw_area_start.x + 1;
    state->draw_h = draw_area_end.y - draw_area_start.y + 1;
    state->display_x = display_area.x;
    state->display_y = display_area.y;
    state->display_w = display_size.x;
    state->display_h = display_size.y;
    return 0;
}

int Psyz_VideoUploadRgb24Frame(const unsigned char* pixels, int w, int h) {
    const int y_offset = (240 - h) / 2;
    if (!pixels || w <= 0 || h <= 0 || w > VRAM_W || h > 240 ||
        !device || !vram_render) {
        return 0;
    }
    SDL_GPUCommandBuffer* cmd = AcquireCmd();
    if (!cmd) return 0;
    u8* map = SDL_MapGPUTransferBuffer(device, tex_upload_transfer, true);
    if (!map) return 0;
    for (int i = 0; i < w * h; i++) {
        map[i * 4 + 0] = pixels[i * 3 + 0];
        map[i * 4 + 1] = pixels[i * 3 + 1];
        map[i * 4 + 2] = pixels[i * 3 + 2];
        map[i * 4 + 3] = 255;
    }
    SDL_UnmapGPUTransferBuffer(device, tex_upload_transfer);
    SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
    const SDL_GPUTextureTransferInfo src = {
        .transfer_buffer = tex_upload_transfer,
        .pixels_per_row = (Uint32)w,
        .rows_per_layer = (Uint32)h,
    };
    for (int page = 0; page < 2; page++) {
        const SDL_GPUTextureRegion dst = {
            .texture = vram_render,
            .x = 0,
            .y = (Uint32)(page * 240 + y_offset),
            .w = (Uint32)w,
            .h = (Uint32)h,
            .d = 1,
        };
        SDL_UploadToGPUTexture(copy, &src, &dst, false);
    }
    SDL_EndGPUCopyPass(copy);
    MarkVramDirty((SDL_Rect){0, y_offset, w, h});
    MarkVramDirty((SDL_Rect){0, 240 + y_offset, w, h});
    SyncNativeVramToScaled(0, y_offset, w, h);
    SyncNativeVramToScaled(0, 240 + y_offset, w, h);
    return 1;
}

static void UpdateScissor() {
    int width = draw_area_end.x - draw_area_start.x + 1;
    int height = draw_area_end.y - draw_area_start.y + 1;

    /* A draw-area change terminates the previous primitive batch even when
     * the new PS1 area is empty.  Keeping the old scissor in that case lets
     * primitives queued for a zero-height area draw over the whole previous
     * viewport when the next valid area is installed. */
    Draw_FlushBuffer();
    if (width <= 0 || height <= 0) {
        scissor_rect.x = draw_area_start.x;
        scissor_rect.y = draw_area_start.y;
        scissor_rect.w = 0;
        scissor_rect.h = 0;
        return;
    }

    if (!sdl3_window || !InitPlatform()) {
        return;
    }
    scissor_rect.x = draw_area_start.x;
    scissor_rect.y = draw_area_start.y;
    scissor_rect.w = width;
    scissor_rect.h = height;
}

unsigned Psyz_VideoGetInternalResolution(void) { return internal_res; }

int Psyz_VideoSetInternalResolution(unsigned multiplier) {
    if (multiplier < 1) {
        return -1;
    }
    if (multiplier > PSYZ_INTERNAL_RES_MAX) {
        WARNF("internal resolution %dx exceeds maximum value of %dx",
              multiplier, PSYZ_INTERNAL_RES_MAX);
        return -1;
    }
    set_internal_res = multiplier;
    if (sdl3_window && is_platform_init_successful) {
        ApplyPendingInternalRes();
    }
    return 0;
}

static void ApplyDisplayPendingChanges() {
    if (!sdl3_window && !InitPlatform()) {
        return;
    }
    ApplyPendingInternalRes();
    if (cur_display_size.x != display_size.x ||
        cur_display_size.y != display_size.y || (!is_window_visible && !headless_window)) {
        if (!is_window_visible && !headless_window) {
            SetWindowLogicalSize(DEFAULT_FRONT_W, DEFAULT_FRONT_H);
        }
        cur_display_size = display_size;
    }
    if (!is_window_visible && !headless_window) {
        SDL_ShowWindow(sdl3_window);
        is_window_visible = true;
    }
    if (cur_disp_horiz != set_disp_horiz || cur_disp_vert != set_disp_vert) {
        cur_disp_horiz = set_disp_horiz;
        cur_disp_vert = set_disp_vert;
    }
}

void Draw_Reset() { NOT_IMPLEMENTED; }

void Draw_DisplayEnable(unsigned int on) {
    disp_on = on;
    if (!on) {
        if (!sdl3_window && !InitPlatform()) {
            return;
        }
    } else {
        ApplyDisplayPendingChanges();
    }
}

void Draw_DisplayArea(unsigned int x, unsigned int y) {
    display_area.x = (int)x;
    display_area.y = (int)y;

    Draw_FlushBuffer();
    ApplyDisplayPendingChanges();
}

void Draw_DisplayHorizontalRange(unsigned int start, unsigned int end) {
    set_disp_horiz = (int)(end - start) / 10;
    ApplyDisplayPendingChanges();
}

void Draw_DisplayVerticalRange(unsigned int start, int unsigned end) {
    set_disp_vert = (int)(end - start);
    ApplyDisplayPendingChanges();
}

void Draw_SetDisplayMode(DisplayMode* mode) {
    // TODO the interlace flag is ignored
    // TODO rgb24 is ignored, the color output will always max the color space
    if (mode->reversed) {
        WARNF("reverse mode not supported");
    }
    is_pal = mode->pal;
    if (mode->horizontal_resolution_368) {
        display_size.x = 368;
    } else {
        switch (mode->horizontal_resolution) {
        case 0:
            display_size.x = 256;
            break;
        case 1:
            display_size.x = 320;
            break;
        case 2:
            display_size.x = 512;
            break;
        case 3:
            display_size.x = 640;
            break;
        default:
            break;
        }
    }
    display_size.y = mode->vertical_resolution ? 480 : 240;
    ApplyDisplayPendingChanges();

    double new_target_fps = mode->pal ? VSYNC_PAL : VSYNC_NTSC;
    if (!target_frame_rate_explicit && new_target_fps != target_frame_rate) {
        UpdateTargetFramerate(new_target_fps);
    }
}

static struct {
    Uint64 correction_ns, gouraud_ns, flush_ns;
    unsigned correction_calls, gouraud_calls, flush_calls;
} dispatch_profile;

static bool DispatchProfileEnabled(void) {
    static int enabled = -1;
    if (enabled < 0) enabled = getenv("PSYZ_GPU_DISPATCH_TRACE") != NULL;
    return enabled != 0;
}

int Draw_ExequeSync() {
    if (DispatchProfileEnabled() &&
        (dispatch_profile.correction_calls || dispatch_profile.flush_calls)) {
        fprintf(stderr, "gpu-dispatch end_ns=%llu correction_ms=%.3f correction_calls=%u gouraud_ms=%.3f gouraud_calls=%u flush_ms=%.3f flush_calls=%u\n",
            (unsigned long long)SDL_GetTicksNS(),
            dispatch_profile.correction_ns / 1000000.0,
            dispatch_profile.correction_calls,
            dispatch_profile.gouraud_ns / 1000000.0,
            dispatch_profile.gouraud_calls,
            dispatch_profile.flush_ns / 1000000.0,
            dispatch_profile.flush_calls);
        memset(&dispatch_profile, 0, sizeof(dispatch_profile));
    }
    return 0;
}

// optimization to avoid sampling the VRAM on an untextured batch draw
static bool batch_has_texture = false;


static bool PsxFlatTriangleSpan(const RasterPoint input[3], int y,
                                int* x_start, int* x_end) {
    RasterPoint v[3] = {input[0], input[1], input[2]};
    for (int i = 1; i < 3; i++) {
        RasterPoint key = v[i];
        int j = i;
        while (j > 0 && v[j - 1].y > key.y) {
            v[j] = v[j - 1];
            j--;
        }
        v[j] = key;
    }
    if (v[2].y == v[0].y || y < v[0].y || y > v[2].y) {
        return false;
    }

    double x1;
    if (y < v[1].y) {
        if (v[1].y == v[0].y) {
            return false;
        }
        double t = (double)(y - v[0].y) / (double)(v[1].y - v[0].y);
        x1 = v[0].x + (v[1].x - v[0].x) * t;
    } else {
        if (v[2].y == v[1].y) {
            return false;
        }
        double t = (double)(y - v[1].y) / (double)(v[2].y - v[1].y);
        x1 = v[1].x + (v[2].x - v[1].x) * t;
    }
    double t = (double)(y - v[0].y) / (double)(v[2].y - v[0].y);
    double x2 = v[0].x + (v[2].x - v[0].x) * t;
    if (x1 > x2) {
        double swap = x1;
        x1 = x2;
        x2 = swap;
    }
    *x_start = (int)ceil(x1);
    *x_end = (int)floor(x2);
    return *x_start <= *x_end;
}

static bool ModernTriangleContains(const PreparedTriangleCoverage *p, int x, int y) {
    return TriangleCoverageContains(p, x, y);
}

static PreparedTextureTriangle PrepareTextureTriangle(const Vertex input[3]) {
    TextureTriangleVertex vertices[3];
    for (int i = 0; i < 3; ++i)
        vertices[i] = (TextureTriangleVertex){
            input[i].x, input[i].y, input[i].u, input[i].v,
            input[i].r, input[i].g, input[i].b};
    return TextureTrianglePrepare(vertices, draw_offset.x, draw_offset.y);
}

static bool PsxTextureTriangleSpan(const PreparedTextureTriangle *triangle,
                                   int y, RasterTextureSpan *span) {
    return TextureTriangleSpanAt(triangle, y, span);
}

static TextureSamplePlane ModernTexturePrepare(const Vertex p[3]) {
    TextureSampleVertex vertices[3];
    for (int i = 0; i < 3; ++i) {
        vertices[i] = (TextureSampleVertex){
            p[i].x + draw_offset.x, p[i].y + draw_offset.y, p[i].u, p[i].v};
    }
    return TextureSamplePrepare(vertices);
}

static bool ModernTextureSample(const TextureSamplePlane *plane, int x, int y,
                                u16* u, u16* v) {
    return TextureSampleAt(plane, x, y, u, v);
}

static bool TextureSamplesDiffer(const Vertex source[4], u16 expected_u,
                                 u16 expected_v, u16 modern_u, u16 modern_v) {
    u8 and_u = source[0].twin & 0xff;
    u8 and_v = (source[0].twin >> 8) & 0xff;
    u8 or_u = (source[0].twin >> 16) & 0xff;
    u8 or_v = (source[0].twin >> 24) & 0xff;
    expected_u = (expected_u & and_u) | or_u;
    expected_v = (expected_v & and_v) | or_v;
    modern_u = (modern_u & and_u) | or_u;
    modern_v = (modern_v & and_v) | or_v;
    return expected_u != modern_u || expected_v != modern_v;
}

static void Draw_EnqueueCompatibilityRun(const Vertex source[4], int x,
                                           int y, int width, bool textured, u16 u,
                                           u16 v, u8 r, u8 g, u8 b) {
    Draw_EnsureBufferWillNotOverflow(4, 6);
    Vertex* q = vertex_cur;
    Vertex pixel = source[0];
    if (textured) {
        pixel.u = u;
        pixel.v = v;
    }
    pixel.r = r;
    pixel.g = g;
    pixel.b = b;
    /* Set common attributes before replication instead of overwriting them
     * independently in all four destination vertices. */
    q[0] = q[1] = q[2] = q[3] = pixel;
    q[0].x = q[2].x = (short)(x - draw_offset.x);
    q[1].x = q[3].x = (short)(x + width - draw_offset.x);
    q[0].y = q[1].y = (short)(y - draw_offset.y);
    q[2].y = q[3].y = (short)(y + 1 - draw_offset.y);
    if (width > 1) q[1].u = q[3].u = u + width;
    index_cur[0] = n_vertices + 0;
    index_cur[1] = n_vertices + 1;
    index_cur[2] = n_vertices + 2;
    index_cur[3] = n_vertices + 1;
    index_cur[4] = n_vertices + 3;
    index_cur[5] = n_vertices + 2;
    Draw_EnqueueBuffer(4, 6);
    batch_saved_vertices += (width - 1) * 4;
    batch_saved_indices += (width - 1) * 6;
}

static void Draw_EnqueueCompatibilityPixel(const Vertex source[4], int x,
        int y, bool textured, u16 u, u16 v, u8 r, u8 g, u8 b) {
    Draw_EnqueueCompatibilityRun(source, x, y, 1, textured, u, v, r, g, b);
}

/* Metal rasterizes a PS1 F4 as two conventional triangles.  The PS1 instead
 * walks inclusive horizontal spans for each triangle; integer rounding can
 * consequently cover pixels outside the mathematical triangle, most visibly
 * on thin rotated quads such as Rage Racer's tachometer needle.  Append only
 * those missing spans, so opaque and semi-transparent pixels already covered
 * by Metal are never drawn twice. */
static void Draw_FillFlatQuadScanlineGaps(const Vertex source[4]) {
    RasterPoint p[4];
    for (int i = 0; i < 4; i++) {
        p[i].x = source[i].x + draw_offset.x;
        p[i].y = source[i].y + draw_offset.y;
    }
    const RasterPoint tri0[3] = {p[0], p[1], p[2]};
    const RasterPoint tri1[3] = {p[1], p[2], p[3]};
    const PreparedTriangleCoverage coverage[2] = {
        TriangleCoveragePrepare(tri0), TriangleCoveragePrepare(tri1)};
    int y_min = p[0].y, y_max = p[0].y;
    for (int i = 1; i < 4; i++) {
        if (p[i].y < y_min) y_min = p[i].y;
        if (p[i].y > y_max) y_max = p[i].y;
    }
    if (y_min < draw_area_start.y) y_min = draw_area_start.y;
    if (y_min < 0) y_min = 0;
    if (y_max > draw_area_end.y) y_max = draw_area_end.y;
    if (y_max >= VRAM_H) y_max = VRAM_H - 1;
    if (y_min > y_max) return;

    for (int y = y_min; y <= y_max; y++) {
        int lo0 = 0, hi0 = 0, lo1 = 0, hi1 = 0;
        bool span0 = PsxFlatTriangleSpan(tri0, y, &lo0, &hi0);
        bool span1 = PsxFlatTriangleSpan(tri1, y, &lo1, &hi1);
        if (!span0 && !span1) continue;
        int candidates[4] = {lo0, hi0, lo1, hi1};
        int candidate_count = (span0 ? 2 : 0) + (span1 ? 2 : 0);
        if (!span0 && span1) {
            candidates[0] = lo1;
            candidates[1] = hi1;
        }
        for (int i = 0; i < candidate_count; i++) {
            int x = candidates[i];
            bool duplicate = false;
            for (int j = 0; j < i; j++) duplicate |= candidates[j] == x;
            if (duplicate || x < draw_area_start.x || x > draw_area_end.x ||
                x < 0 || x >= VRAM_W) continue;
            bool expected =
                ((span0 && x >= lo0 && x <= hi0) ||
                 (span1 && x >= lo1 && x <= hi1));
            bool covered = ModernTriangleContains(&coverage[0], x, y) ||
                           ModernTriangleContains(&coverage[1], x, y);
            if (expected && !covered)
                Draw_EnqueueCompatibilityPixel(source, x, y, false, 0, 0,
                                               source[0].r, source[0].g,
                                               source[0].b);
        }
    }
}

static void Draw_FillTexturedQuadScanlineGapsInternal(const Vertex source[4],
                                              bool gouraud) {
    RasterPoint p[4];
    for (int i = 0; i < 4; i++) {
        p[i].x = source[i].x + draw_offset.x;
        p[i].y = source[i].y + draw_offset.y;
    }
    const RasterPoint points0[3] = {p[0], p[1], p[2]};
    const RasterPoint points1[3] = {p[1], p[2], p[3]};
    const PreparedTriangleCoverage coverage[2] = {
        TriangleCoveragePrepare(points0), TriangleCoveragePrepare(points1)};
    const Vertex vertices0[3] = {source[0], source[1], source[2]};
    const Vertex vertices1[3] = {source[1], source[2], source[3]};
    const PreparedTextureTriangle raster[2] = {
        PrepareTextureTriangle(vertices0), PrepareTextureTriangle(vertices1)};
    const TextureSamplePlane planes[2] = {
        ModernTexturePrepare(vertices0), ModernTexturePrepare(vertices1)};
    bool correct_all = !gouraud && TextureSampleAlwaysUnstable(&planes[0]) &&
        TextureSampleAlwaysUnstable(&planes[1]);
    int y_min = p[0].y, y_max = p[0].y;
    for (int i = 1; i < 4; i++) {
        if (p[i].y < y_min) y_min = p[i].y;
        if (p[i].y > y_max) y_max = p[i].y;
    }
    if (y_min < draw_area_start.y) y_min = draw_area_start.y;
    if (y_min < 0) y_min = 0;
    if (y_max > draw_area_end.y) y_max = draw_area_end.y;
    if (y_max >= VRAM_H) y_max = VRAM_H - 1;
    if (y_min > y_max) return;

    for (int y = y_min; y <= y_max; y++) {
        RasterTextureSpan span0 = {0}, span1 = {0};
        bool has0 = PsxTextureTriangleSpan(&raster[0], y, &span0);
        bool has1 = PsxTextureTriangleSpan(&raster[1], y, &span1);
        if (!has0 && !has1) continue;
        PreparedTextureSpan samples[2] = {{0}, {0}};
        if (has0) samples[0] = TextureSpanPrepare(&span0);
        if (has1) samples[1] = TextureSpanPrepare(&span1);
        int x_min = has0 ? span0.x_start : span1.x_start;
        int x_max = has0 ? span0.x_end : span1.x_end;
        if (has1) {
            if (span1.x_start < x_min) x_min = span1.x_start;
            if (span1.x_end > x_max) x_max = span1.x_end;
        }
        if (x_min < draw_area_start.x) x_min = draw_area_start.x;
        if (x_min < 0) x_min = 0;
        if (x_max > draw_area_end.x) x_max = draw_area_end.x;
        if (x_max >= VRAM_W) x_max = VRAM_W - 1;
        int candidates[4] = {span0.x_start, span0.x_end,
                             span1.x_start, span1.x_end};
        int candidate_count = (has0 ? 2 : 0) + (has1 ? 2 : 0);
        if (!has0 && has1) {
            candidates[0] = span1.x_start;
            candidates[1] = span1.x_end;
        }
        int iterations = gouraud ? candidate_count : x_max - x_min + 1;
        TriangleCoverageSpan covered_rows[2] = {{1, 0}, {1, 0}};
        /* Division per row pays off for longer spans. Keep the point test
         * for short rows and Gouraud's at-most-four endpoint candidates. */
        bool row_coverage = !gouraud && !correct_all && x_max - x_min >= 15;
        if (row_coverage) {
            covered_rows[0] = TriangleCoverageRow(&coverage[0], y, x_min, x_max);
            covered_rows[1] = TriangleCoverageRow(&coverage[1], y, x_min, x_max);
        }
        for (int i = 0; i < iterations; i++) {
            int x = gouraud ? candidates[i] : x_min + i;
            if (x < x_min || x > x_max) continue;
            if (gouraud) {
                bool duplicate = false;
                for (int j = 0; j < i; j++) duplicate |= candidates[j] == x;
                if (duplicate) continue;
            }
            bool expected0 = has0 && x >= span0.x_start && x <= span0.x_end;
            bool expected1 = has1 && x >= span1.x_start && x <= span1.x_end;
            bool covered0 = !correct_all && (!row_coverage ? ModernTriangleContains(&coverage[0], x, y)
                : x >= covered_rows[0].first && x <= covered_rows[0].last);
            bool covered1 = !correct_all && (!row_coverage ? ModernTriangleContains(&coverage[1], x, y)
                : x >= covered_rows[1].first && x <= covered_rows[1].last);
            if (!expected0 && !expected1) continue;
            /* Gouraud correction only fills uncovered endpoints; covered
             * pixels need neither a UV comparison nor colour interpolation. */
            if (gouraud && (covered0 || covered1)) continue;

            /* The PS1 submits triangle 0 followed by triangle 1. If their
             * inclusive spans overlap, triangle 1 owns the final texel. */
            const PreparedTextureSpan* sample = &samples[expected1 ? 1 : 0];
            u16 expected_u, expected_v;
            u8 expected_r, expected_g, expected_b;
            TextureSpanSampleUV(sample, x, &expected_u, &expected_v);
            bool correct = !covered0 && !covered1;
            if (!correct && !gouraud) {
                const TextureSamplePlane* modern = &planes[covered1 ? 1 : 0];
                u16 modern_u, modern_v;
                bool unstable = ModernTextureSample(modern, x, y,
                                                     &modern_u, &modern_v);
                correct = unstable ||
                    TextureSamplesDiffer(source, expected_u, expected_v,
                                         modern_u, modern_v);
            }
            if (correct) {
                TextureSpanSampleColor(sample, x,
                                       &expected_r, &expected_g, &expected_b);
                int run = 1;
                static int reference = -1;
                if (reference < 0) reference = getenv("PSYZ_REFERENCE_CORRECTION_PIXELS") != NULL;
                if (!reference && correct_all) {
                    int end = expected1 ? span1.x_end : span0.x_end;
                    if (!expected1 && has1 && span1.x_start > x && end >= span1.x_start)
                        end = span1.x_start - 1;
                    if (end > x_max) end = x_max;
                    run = TextureSpanConsecutiveRun(sample, x, expected_u, end - x + 1);
                }
                /* Do not move the original per-pixel flush boundary: later
                 * primitives can sample pixels written by an earlier batch. */
                Draw_EnsureBufferWillNotOverflow(4, 6);
                int available = (MAX_VERTEX_COUNT - n_vertices - batch_saved_vertices) / 4;
                int index_available = (MAX_INDEX_COUNT - n_indices - batch_saved_indices) / 6;
                if (available > index_available) available = index_available;
                if (available < 1) return; /* A failed GPU flush left no space. */
                if (run > available) run = available;
                Draw_EnqueueCompatibilityRun(source, x, y, run, true,
                                               expected_u, expected_v,
                                               expected_r, expected_g,
                                               expected_b);
                i += run - 1;
            }
        }
    }
}

static void Draw_FillTexturedQuadScanlineGaps(const Vertex source[4], bool gouraud) {
    bool profile = DispatchProfileEnabled();
    Uint64 start = profile ? SDL_GetTicksNS() : 0;
    Draw_FillTexturedQuadScanlineGapsInternal(source, gouraud);
    if (profile) {
        Uint64 elapsed = SDL_GetTicksNS() - start;
        dispatch_profile.correction_ns += elapsed;
        dispatch_profile.correction_calls++;
        if (gouraud) {
            dispatch_profile.gouraud_ns += elapsed;
            dispatch_profile.gouraud_calls++;
        }
    }
}

static void Draw_CorrectOpaqueSpriteSamples(const Vertex source[4]) {
    TextureSampleVertex sprite[4];
    for (int i = 0; i < 4; ++i)
        sprite[i] = (TextureSampleVertex){
            source[i].x, source[i].y, source[i].u, source[i].v};
    const bool unit_sprite = TextureSampleIsUnitSprite(sprite);
    const Vertex triangles[2][3] = {
        {source[0], source[1], source[2]},
        {source[1], source[3], source[2]},
    };
    RasterPoint points[2][3];
    const TextureSamplePlane planes[2] = {
        ModernTexturePrepare(triangles[0]), ModernTexturePrepare(triangles[1])};
    for (int triangle = 0; triangle < 2; triangle++) {
        for (int vertex = 0; vertex < 3; vertex++) {
            points[triangle][vertex].x =
                triangles[triangle][vertex].x + draw_offset.x;
            points[triangle][vertex].y =
                triangles[triangle][vertex].y + draw_offset.y;
        }
    }
    const PreparedTriangleCoverage coverage[2] = {
        TriangleCoveragePrepare(points[0]), TriangleCoveragePrepare(points[1])};
    const int x0 = source[0].x + draw_offset.x;
    const int y0 = source[0].y + draw_offset.y;
    const int x1 = source[3].x + draw_offset.x;
    const int y1 = source[3].y + draw_offset.y;
    for (int y = y0; y < y1; y++) {
        if (y < draw_area_start.y || y > draw_area_end.y ||
            y < 0 || y >= VRAM_H) continue;
        for (int x = x0; x < x1; x++) {
            if (x < draw_area_start.x || x > draw_area_end.x ||
                x < 0 || x >= VRAM_W) continue;
            u16 expected_u = (u16)(source[0].u + x - x0);
            u16 expected_v = (u16)(source[0].v + y - y0);
            bool correct = unit_sprite;
            if (!correct) {
                int triangle = ModernTriangleContains(&coverage[1], x, y) ? 1 : 0;
                bool covered = ModernTriangleContains(&coverage[triangle], x, y);
                u16 modern_u = 0, modern_v = 0;
                bool unstable = covered && ModernTextureSample(
                    &planes[triangle], x, y, &modern_u, &modern_v);
                correct = !covered || unstable || TextureSamplesDiffer(
                    source, expected_u, expected_v, modern_u, modern_v);
            }
            if (correct) {
                Draw_EnqueueCompatibilityPixel(
                    source, x, y, true, expected_u, expected_v,
                    source[0].r, source[0].g, source[0].b);
            }
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
    u16 tpage = -1, clut = -1, pad2, pad3;
    Vertex* v;

    /* writePacket stores a Gouraud packet's following vertex color through
     * v + 1.  On the final vertex that is a parser scratch write beyond the
     * four vertices eventually enqueued, so reserve one extra slot. */
    Draw_EnsureBufferWillNotOverflow(isGouraud ? 9 : 8, 12);
    v = vertex_cur;
    if (isShadeTex) {
        v->r = (unsigned char)(*packets >> 0);
        v->g = (unsigned char)(*packets >> 8);
        v->b = (unsigned char)(*packets >> 16);
    } else {
        v->r = v->g = v->b = 0x80;
    }
    v->a = code & SEMITRANSP ? 0x80 : 0xFF;
    packets++;
    len--;
    if (isPoly) {
        if (code & TRIANGLE) {
            int wr, nVertices, nIndices;
            wr = writePacket(v++, code, len, packets, &clut);
            packets += wr;
            len -= wr;
            wr = writePacket(v++, code, len, packets, &tpage);
            packets += wr;
            len -= wr;
            wr = writePacket(v++, code, len, packets, &pad2);
            packets += wr;
            len -= wr;

            index_cur[0] = n_vertices + 0;
            index_cur[1] = n_vertices + 1;
            index_cur[2] = n_vertices + 2;
            if (code & EXTRA_VERTEX) {
                index_cur[3] = n_vertices + 1;
                index_cur[4] = n_vertices + 3;
                index_cur[5] = n_vertices + 2;
                nVertices = 4;
                nIndices = 6;
                wr = writePacket(v, code, len, packets, &pad3);
                packets += wr;
                len -= wr;
            } else {
                nVertices = 3;
                nIndices = 3;
            }
            // HACK last rgb are not read by writePacket, so we patch the amount
            if (isGouraud) {
                packets--;
                len++;
            }

            if (isTextured) {
                /* The GPU latches the texture page carried by a textured
                 * polygon. Later sprites have no tpage field and consume
                 * that latched value. */
                cur_tpage = tpage;
            } else {
                clut = -1;
                tpage = cur_tpage | TPAGE_NOTEXTURE;
            }
            if (CanPolyDither(isGouraud, isTextured, isShadeTex)) {
                tpage |= TPAGE_DITHER;
            }
            if (!isGouraud || !isShadeTex) {
                VRGBA(vertex_cur[1]) = VRGBA(vertex_cur[2]) =
                    VRGBA(vertex_cur[3]) = VRGBA(vertex_cur[0]);
            }

            SET_TC_ALL(vertex_cur, tpage, clut);
            TraceGpuPrimitive(vertex_cur, nVertices, code, tpage, clut,
                              draw_offset.x,
                              draw_offset.y - (draw_area_start.y / 240) * 240,
                              draw_area_start.x, draw_area_start.y,
                              draw_area_end.x, draw_area_end.y);
            Vertex compatibility_quad[4];
            bool fill_quad_gaps = nVertices == 4 && !(code & SEMITRANSP);
            if (fill_quad_gaps) {
                memcpy(compatibility_quad, vertex_cur,
                       sizeof(compatibility_quad));
            }
            batch_has_texture |= isTextured;
            Draw_EnqueueBuffer(nVertices, nIndices);
            if (fill_quad_gaps) {
                if (isTextured) {
                    Draw_FillTexturedQuadScanlineGaps(compatibility_quad,
                                                      isGouraud);
                } else {
                    Draw_FillFlatQuadScanlineGaps(compatibility_quad);
                }
            }
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

        // accumulate line points first, then convert them to triangles
        short px[4], py[4];
        unsigned char cr[4], cg[4], cb[4], ca[4];
        int parsed = 0;
        cr[0] = v->r;
        cg[0] = v->g;
        cb[0] = v->b;
        ca[0] = v->a;
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
                ca[i + 1] = code & SEMITRANSP ? 0x80 : 0xFF;
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
        if (nSegments >= 1) {
            // for LINE_*4: 12 verts, 18 indices
            Draw_EnsureBufferWillNotOverflow(nSegments * 4, nSegments * 6);
            for (int s = 0; s < nSegments; s++) {
                short x0 = px[s];
                short y0 = py[s];
                short x1 = px[s + 1];
                short y1 = py[s + 1];
                int dx = x1 - x0;
                int dy = y1 - y0;

                // decides to thicken horizontally or vertically
                short ox, oy, ex, ey;
                if ((dx < 0 ? -dx : dx) >= (dy < 0 ? -dy : dy)) {
                    ox = 0;
                    oy = 1;
                    ex = dx >= 0 ? 1 : -1;
                    ey = 0;
                } else {
                    ox = 1;
                    oy = 0;
                    ex = 0;
                    ey = dy >= 0 ? 1 : -1;
                }

                Vertex* q = vertex_cur;
                unsigned short base = n_vertices;
                q[0].x = x0;
                q[0].y = y0;
                q[1].x = (short)(x1 + ex);
                q[1].y = (short)(y1 + ey);
                q[2].x = (short)(x1 + ex + ox);
                q[2].y = (short)(y1 + ey + oy);
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
                u16 lt = cur_tpage | TPAGE_NOTEXTURE;
                if (CanLineDither()) {
                    lt |= TPAGE_DITHER;
                }
                for (int k = 0; k < 4; k++) {
                    SET_TC(&q[k], lt, -1);
                }
                index_cur[0] = base + 0;
                index_cur[1] = base + 1;
                index_cur[2] = base + 2;
                index_cur[3] = base + 0;
                index_cur[4] = base + 2;
                index_cur[5] = base + 3;
                Draw_EnqueueBuffer(4, 6);
            }
        }
    } else if (isTile) {
        int x, y, w, h, tu, tv;
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
            tpage = cur_tpage | TPAGE_NOTEXTURE;
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
            // TODO warn about unrecognized code
            break;
        }
        vertex_cur[0].x = (short)(x);
        vertex_cur[0].y = (short)(y);
        vertex_cur[1].x = (short)(x + w);
        vertex_cur[1].y = (short)(y);
        vertex_cur[2].x = (short)(x);
        vertex_cur[2].y = (short)(y + h);
        vertex_cur[3].x = (short)(x + w);
        vertex_cur[3].y = (short)(y + h);
        if (isTextured) {
            vertex_cur[0].u = tu;
            vertex_cur[0].v = tv;
            vertex_cur[1].u = tu + w;
            vertex_cur[1].v = tv;
            vertex_cur[2].u = tu;
            vertex_cur[2].v = tv + h;
            vertex_cur[3].u = tu + w;
            vertex_cur[3].v = tv + h;
        }
        VRGBA(vertex_cur[1]) = VRGBA(vertex_cur[2]) = VRGBA(vertex_cur[3]) =
            VRGBA(vertex_cur[0]);
        index_cur[0] = n_vertices + 0;
        index_cur[1] = n_vertices + 1;
        index_cur[2] = n_vertices + 2;
        index_cur[3] = n_vertices + 1;
        index_cur[4] = n_vertices + 3;
        index_cur[5] = n_vertices + 2;
        SET_TC_ALL(vertex_cur, tpage, clut);
        Vertex compatibility_quad[4];
        bool correct_sprite_uv = isTextured && !(code & SEMITRANSP);
        if (correct_sprite_uv) {
            memcpy(compatibility_quad, vertex_cur,
                   sizeof(compatibility_quad));
        }
        TraceGpuPrimitive(vertex_cur, 4, code, tpage, clut,
                          draw_offset.x,
                          draw_offset.y - (draw_area_start.y / 240) * 240,
                          draw_area_start.x, draw_area_start.y,
                          draw_area_end.x, draw_area_end.y);
        Draw_EnqueueBuffer(4, 6);
        /* SPRT samples one texel per destination pixel; host triangle
         * interpolation can land an integral UV a float ULP below its texel
         * boundary, most visibly on wide title sprites. */
        if (correct_sprite_uv) {
            Draw_CorrectOpaqueSpriteSamples(compatibility_quad);
        }
    }
    batch_has_texture |= isTextured;
    return max_len - len;
}

void Draw_SetAreaStart(int x, int y) {
    if (getenv("RAGE_GPU_TRACE_AREA") != NULL) {
        fprintf(stderr,
                "gpu-area-start next=%d,%d active=%d,%d,%d,%d pending=%d\n",
                x, y, scissor_rect.x, scissor_rect.y, scissor_rect.w,
                scissor_rect.h, n_vertices);
    }
    draw_area_start.x = x;
    draw_area_start.y = y;
}
void Draw_SetAreaEnd(int x, int y) {
    if (getenv("RAGE_GPU_TRACE_AREA") != NULL) {
        fprintf(stderr,
                "gpu-area-end next=%d,%d start=%d,%d active=%d,%d,%d,%d "
                "pending=%d\n",
                x, y, draw_area_start.x, draw_area_start.y, scissor_rect.x,
                scissor_rect.y, scissor_rect.w, scissor_rect.h, n_vertices);
    }
    draw_area_end.x = x;
    draw_area_end.y = y;
    UpdateScissor();
}
void Draw_SetOffset(int x, int y) {
    Draw_FlushBuffer();

    x = x % VRAM_W;
    y = y % VRAM_H;
    if (x < 0) {
        x += VRAM_W;
    }
    if (y < 0) {
        y += VRAM_H;
    }
    draw_offset.x = x;
    draw_offset.y = y;
}

void Draw_ClearImage(PS1_RECT* rect, u_char r, u_char g, u_char b) {
    if (rect->w == 0 || rect->h == 0) {
        return;
    }
    if (!sdl3_window && !InitPlatform()) {
        return;
    }
    SDL_GPUCommandBuffer* cmd = AcquireCmd();
    if (!cmd) {
        return;
    }

    Vertex quad[4] = {0};
    quad[0].x = rect->x;
    quad[0].y = rect->y;
    quad[1].x = (short)(rect->x + rect->w);
    quad[1].y = rect->y;
    quad[2].x = rect->x;
    quad[2].y = (short)(rect->y + rect->h);
    quad[3].x = (short)(rect->x + rect->w);
    quad[3].y = (short)(rect->y + rect->h);
    for (int i = 0; i < 4; i++) {
        quad[i].r = r;
        quad[i].g = g;
        quad[i].b = b;
        quad[i].a = 0;
    }

    Vertex* map = SDL_MapGPUTransferBuffer(device, vtx_transfer, true);
    if (!map) {
        ERRORF("SDL_MapGPUTransferBuffer: %s", SDL_GetError());
        return;
    }
    memcpy(map, quad, sizeof(quad));
    SDL_UnmapGPUTransferBuffer(device, vtx_transfer);

    SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
    const SDL_GPUTransferBufferLocation src = {.transfer_buffer = vtx_transfer};
    const SDL_GPUBufferRegion dst = {
        .buffer = clear_vbuf, .size = sizeof(quad)};
    SDL_UploadToGPUBuffer(copy, &src, &dst, true);
    SDL_EndGPUCopyPass(copy);

    const SDL_GPUColorTargetInfo target = {
        .texture = vram_render,
        .load_op = SDL_GPU_LOADOP_LOAD,
        .store_op = SDL_GPU_STOREOP_STORE,
    };
    SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &target, 1, NULL);
    SDL_BindGPUGraphicsPipeline(pass, pipe_clear);
    const SDL_GPUBufferBinding vb = {.buffer = clear_vbuf};
    SDL_BindGPUVertexBuffers(pass, 0, &vb, 1);
    SDL_DrawGPUPrimitives(pass, 4, 1, 0, 0);
    SDL_EndGPURenderPass(pass);
    MarkVramDirty((SDL_Rect){rect->x, rect->y, rect->w, rect->h});
    SyncNativeVramToScaled(rect->x, rect->y, rect->w, rect->h);
}

void Draw_LoadImage(PS1_RECT* rect, u_long* p) {
    if (rect->w == 0 || rect->h == 0) {
        return;
    }
    if (!sdl3_window && !InitPlatform()) {
        return;
    }
    SDL_GPUCommandBuffer* cmd = AcquireCmd();
    if (!cmd) {
        return;
    }
    u8* map = SDL_MapGPUTransferBuffer(device, tex_upload_transfer, true);
    if (!map) {
        ERRORF("SDL_MapGPUTransferBuffer: %s", SDL_GetError());
        return;
    }
    ConvertRgb5551ToRgba8888((const u16*)p, map, (size_t)rect->w * rect->h);
    SDL_UnmapGPUTransferBuffer(device, tex_upload_transfer);

    SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
    const SDL_GPUTextureTransferInfo src = {
        .transfer_buffer = tex_upload_transfer,
        .pixels_per_row = rect->w,
        .rows_per_layer = rect->h,
    };
    const SDL_GPUTextureRegion dst = {
        .texture = texture_upload_context == 1 ? auxiliary_vram : vram_render,
        .x = (Uint32)rect->x,
        .y = (Uint32)rect->y,
        .w = rect->w,
        .h = rect->h,
        .d = 1,
    };
    SDL_UploadToGPUTexture(copy, &src, &dst, false);
    SDL_EndGPUCopyPass(copy);
    if (texture_upload_context == 0) {
        MarkVramDirty((SDL_Rect){rect->x, rect->y, rect->w, rect->h});
        VramReadCacheWrite(&vram_read_cache, rect->x, rect->y, rect->w,
                           rect->h, (const u16 *)p);
        SyncNativeVramToScaled(rect->x, rect->y, rect->w, rect->h);
    }
}

void Psyz_GpuTextureUploadContext(int context) {
    int next = context == 1 ? 1 : 0;

    if (next == 1 && texture_upload_context != 1 && device &&
        vram_render && auxiliary_vram) {
        Draw_FlushBuffer();
        SDL_GPUCommandBuffer* cmd = AcquireCmd();
        if (cmd) {
            const SDL_GPUBlitInfo blit = {
                .source = {
                    .texture = vram_render,
                    .w = VRAM_W,
                    .h = VRAM_H,
                },
                .destination = {
                    .texture = auxiliary_vram,
                    .w = VRAM_W,
                    .h = VRAM_H,
                },
                .load_op = SDL_GPU_LOADOP_DONT_CARE,
                .filter = SDL_GPU_FILTER_NEAREST,
            };
            SDL_BlitGPUTexture(cmd, &blit);
        }
    }
    texture_upload_context = next;
}

void Draw_StoreImage(PS1_RECT* rect, u_long* p) {
    if (rect->w == 0 || rect->h == 0) {
        return;
    }
    if (!device || !vram_render) {
        return;
    }
    Draw_FlushBuffer(); // flush primitives before operating with the VRAM
    bool cached = VramReadCacheRead(&vram_read_cache, rect->x, rect->y,
                                   rect->w, rect->h, (u16 *)p);
    /* Diagnostic oracle: force the original GPU readback on cache hits. */
    static int verify_cache = -1;
    if (verify_cache < 0)
        verify_cache = getenv("PSYZ_VERIFY_VRAM_READ_CACHE") != NULL;
    if (cached && !verify_cache) return;

    const size_t pixels = (size_t)rect->w * rect->h;
    u8* rgba = malloc(pixels * 4);
    if (!rgba) {
        return;
    }
    if (DownloadVramRegionAsRGBA8888(
            rect->x, rect->y, rect->w, rect->h, rgba)) {
        if (cached) {
            ConvertRgba8888ToRgb5551(rgba, (u16 *)rgba, pixels);
            bool matches = memcmp(p, rgba, pixels * sizeof(u16)) == 0;
            fprintf(stderr, "vram-read-cache verify=%s rect=%d,%d,%d,%d\n",
                    matches ? "match" : "MISMATCH",
                    rect->x, rect->y, rect->w, rect->h);
            memcpy(p, rgba, pixels * sizeof(u16));
        } else {
            ConvertRgba8888ToRgb5551(rgba, (u16*)p, pixels);
        }
        VramReadCacheWrite(&vram_read_cache, rect->x, rect->y, rect->w, rect->h,
                          (const u16 *)p);
    }
    free(rgba);
}

void Draw_MoveImage(PS1_RECT* rect, unsigned int x, unsigned int y) {
    if (rect->x == x && rect->y == y) {
        return;
    }
    Draw_FlushBuffer(); // flush primitives before operating with the VRAM

    int src_x = CLAMP(rect->x, 0, VRAM_W - 1);
    int src_y = CLAMP(rect->y, 0, VRAM_H - 1);
    int src_w = CLAMP(rect->w, 0, VRAM_W - src_x);
    int src_h = CLAMP(rect->h, 0, VRAM_H - src_y);
    int dst_x = CLAMP((int)x, 0, VRAM_W - 1);
    int dst_y = CLAMP((int)y, 0, VRAM_H - 1);
    int copy_w = CLAMP(src_w, 0, VRAM_W - dst_x);
    int copy_h = CLAMP(src_h, 0, VRAM_H - dst_y);

    if (src_x != rect->x || src_y != rect->y || src_w != rect->w ||
        src_h != rect->h || dst_x != (int)x || dst_y != (int)y ||
        copy_w != src_w || copy_h != src_h) {
        WARNF("params out of bounds: src=(%d,%d,%d,%d)->(%d,%d,%d,%d) "
              "dst=(%d,%d)->(%d,%d)",
              rect->x, rect->y, rect->w, rect->h, src_x, src_y, src_w, src_h,
              (int)x, (int)y, dst_x, dst_y);
    }

    if (copy_w <= 0 || copy_h <= 0) {
        WARNF("nothing to copy");
        return;
    }

    SDL_GPUCommandBuffer* cmd = AcquireCmd();
    if (!cmd) {
        return;
    }
    SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
    if (!RectsOverlap(src_x, src_y, dst_x, dst_y, copy_w, copy_h)) {
        const SDL_GPUTextureLocation src = {
            .texture = vram_render, .x = (Uint32)src_x, .y = (Uint32)src_y};
        const SDL_GPUTextureLocation bounce = {
            .texture = vram_sample, .x = (Uint32)src_x, .y = (Uint32)src_y};
        const SDL_GPUTextureLocation dst = {
            .texture = vram_render, .x = (Uint32)dst_x, .y = (Uint32)dst_y};
        SDL_CopyGPUTextureToTexture(
            copy, &src, &bounce, (Uint32)copy_w, (Uint32)copy_h, 1, false);
        SDL_CopyGPUTextureToTexture(
            copy, &bounce, &dst, (Uint32)copy_w, (Uint32)copy_h, 1, false);
    } else {
        // Slow-path, simulating real hardware behaviour
        // This behaviour is tested via move_image_overlap
        for (int row = 0; row < copy_h; row++) {
            const SDL_GPUTextureLocation src = {.texture = vram_render,
                                                .x = (Uint32)src_x,
                                                .y = (Uint32)(src_y + row)};
            const SDL_GPUTextureLocation dst = {.texture = vram_render,
                                                .x = (Uint32)dst_x,
                                                .y = (Uint32)(dst_y + row)};
            SDL_CopyGPUTextureToTexture(
                copy, &src, &dst, (Uint32)copy_w, 1, 1, false);
        }
    }
    SDL_EndGPUCopyPass(copy);
    MarkVramDirty((SDL_Rect){dst_x, dst_y, copy_w, copy_h});
    MarkVramDirty((SDL_Rect){src_x, src_y, copy_w, copy_h});
    SyncNativeVramToScaled(dst_x, dst_y, copy_w, copy_h);
}

void Draw_ResetBuffer(void) {
    n_vertices = 0;
    n_indices = 0;
    batch_saved_vertices = batch_saved_indices = 0;
    vertex_cur = vertex_buf;
    index_cur = index_buf;
    batch_has_texture = false;
}

static bool BatchNeedsVramCopy(void) {
    static int reference = -1;
    if (reference < 0) reference = getenv("PSYZ_REFERENCE_FULL_VRAM_BATCH") != NULL;
    if (reference || external_vram_handle) return true;
    uint32_t previous = 0;
    bool have_previous = false;
    for (unsigned i = 0; i < n_vertices; ++i) {
        const Vertex *vertex = &vertex_buf[i];
        if (vertex->t & TPAGE_NOTEXTURE) continue;
        uint32_t key = ((uint32_t)vertex->t << 16) | vertex->c;
        if (have_previous && previous == key) continue;
        previous = key;
        have_previous = true;
        if (VramSampleMaskTextureDirty(&sample_dirty, vertex->t, vertex->c)) return true;
    }
    return false;
}

static void Draw_FlushBufferInternal(void) {
    if (n_vertices == 0) {
        return;
    }
    if (scissor_rect.w <= 0 || scissor_rect.h <= 0) {
        Draw_ResetBuffer();
        return;
    }
    SDL_GPUCommandBuffer* cmd = AcquireCmd();
    if (!cmd) {
        return;
    }

    u8* map = SDL_MapGPUTransferBuffer(device, vtx_transfer, true);
    if (!map) {
        ERRORF("SDL_MapGPUTransferBuffer: %s", SDL_GetError());
        return;
    }
    const Uint32 vtx_size = (Uint32)(sizeof(Vertex) * n_vertices);
    const Uint32 idx_size = (Uint32)(sizeof(*index_buf) * n_indices);
    const Uint32 idx_offset = MAX_VERTEX_COUNT * sizeof(Vertex);
    memcpy(map, vertex_buf, vtx_size);
    memcpy(map + idx_offset, index_buf, idx_size);
    SDL_UnmapGPUTransferBuffer(device, vtx_transfer);

    SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
    const SDL_GPUTransferBufferLocation vtx_src = {
        .transfer_buffer = vtx_transfer};
    const SDL_GPUBufferRegion vtx_dst = {.buffer = vbuf, .size = vtx_size};
    SDL_UploadToGPUBuffer(copy, &vtx_src, &vtx_dst, true);
    const SDL_GPUTransferBufferLocation idx_src = {
        .transfer_buffer = vtx_transfer, .offset = idx_offset};
    const SDL_GPUBufferRegion idx_dst = {.buffer = ibuf, .size = idx_size};
    SDL_UploadToGPUBuffer(copy, &idx_src, &idx_dst, true);
    bool copy_vram = batch_has_texture && BatchNeedsVramCopy();
    static int trace_vram = -1;
    if (trace_vram < 0) trace_vram = getenv("PSYZ_VRAM_BATCH_TRACE") != NULL;
    if (trace_vram && batch_has_texture)
        fprintf(stderr, "vram-batch-sync copy=%d bytes=%u vertices=%u\n",
                copy_vram, copy_vram ? VRAM_W * VRAM_H * 4u : 0u, n_vertices);
    if (copy_vram) {
        /* vram_render is both the framebuffer and PS1 VRAM, while shaders must
         * sample a separate texture to avoid a render-target feedback loop.
         * Copy the complete mirror whenever a sampled page/CLUT may be stale.
         * Dirty tiles survive unrelated batches and read-cache operations;
         * only a complete mirror copy clears them. No partial texture copy
         * or resource cycling can discard unchanged pages. */
        const SDL_GPUTextureLocation vram_src = {.texture = vram_render};
        const SDL_GPUTextureLocation vram_dst = {.texture = vram_sample};
        SDL_CopyGPUTextureToTexture(
            copy, &vram_src, &vram_dst, VRAM_W, VRAM_H, 1, false);
        VramSampleMaskReset(&sample_dirty, false);
    }
    SDL_EndGPUCopyPass(copy);

    const SDL_GPUColorTargetInfo target = {
        .texture = GetRenderTarget(),
        .load_op = SDL_GPU_LOADOP_LOAD,
        .store_op = SDL_GPU_STOREOP_STORE,
    };
    SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &target, 1, NULL);
    SDL_Rect scaled_scissor = {
        scissor_rect.x * (int)internal_res, scissor_rect.y * (int)internal_res,
        scissor_rect.w * (int)internal_res, scissor_rect.h * (int)internal_res};
    SDL_SetGPUScissor(pass, &scaled_scissor);
    const SDL_GPUBufferBinding vb = {.buffer = vbuf};
    SDL_BindGPUVertexBuffers(pass, 0, &vb, 1);
    const SDL_GPUBufferBinding ib = {.buffer = ibuf};
    SDL_BindGPUIndexBuffer(pass, &ib, SDL_GPU_INDEXELEMENTSIZE_16BIT);
    const float offset_ubo[4] = {
        (float)draw_offset.x, (float)draw_offset.y, 0, 0};
    SDL_PushGPUVertexUniformData(cmd, 0, offset_ubo, sizeof(offset_ubo));

    // every primitive (including lines, expanded to quads) is a triangle list
    const int prim_size = 3;
    int start = 0;
    bool cur_subtract = false;
    bool cur_auxiliary = false;
    bool pipeline_bound = false;
    bool sampler_bound = false;
    while (start < n_indices) {
        Vertex* v = &vertex_buf[index_buf[start]];
        bool need_subtract = is_subtract_abr(v);
        bool need_auxiliary = (v->t & TPAGE_AUX_TEXTURE) != 0;
        int end = start + prim_size;
        while (end < n_indices) {
            v = &vertex_buf[index_buf[end]];
            bool next_subtract = is_subtract_abr(v);
            bool next_auxiliary = (v->t & TPAGE_AUX_TEXTURE) != 0;
            if (next_subtract != need_subtract ||
                next_auxiliary != need_auxiliary) {
                break;
            }
            end += prim_size;
        }
        if (!sampler_bound || need_auxiliary != cur_auxiliary) {
            const SDL_GPUTextureSamplerBinding binding = {
                .texture = need_auxiliary ? auxiliary_vram : vram_sample,
                .sampler = vram_sampler};
            SDL_BindGPUFragmentSamplers(pass, 0, &binding, 1);
            cur_auxiliary = need_auxiliary;
            sampler_bound = true;
        }
        if (!pipeline_bound || need_subtract != cur_subtract) {
            SDL_GPUGraphicsPipeline* pipe =
                need_subtract ? pipe_tri_sub : pipe_tri_add;
            SDL_BindGPUGraphicsPipeline(pass, pipe);
            cur_subtract = need_subtract;
            pipeline_bound = true;
        }
        SDL_DrawGPUIndexedPrimitives(
            pass, (Uint32)(end - start), 1, (Uint32)start, 0, 0);
        start = end;
    }
    SDL_EndGPURenderPass(pass);
    if (internal_res <= 1) {
        MarkVramDirty(scissor_rect);
    }
    SyncScaledVramToNative();
    Draw_ResetBuffer();
}

void Draw_FlushBuffer(void) {
    bool profile = n_vertices != 0 && DispatchProfileEnabled();
    Uint64 start = profile ? SDL_GetTicksNS() : 0;
    Draw_FlushBufferInternal();
    if (profile) {
        dispatch_profile.flush_ns += SDL_GetTicksNS() - start;
        dispatch_profile.flush_calls++;
    }
}
