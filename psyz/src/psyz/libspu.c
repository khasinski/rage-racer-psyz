#include <psyz.h>
#include <libspu.h>
#include <psyz/log.h>
#include "../../decomp/src/libspu/libspu_private.h"

static SPU_RXX spu_RXX;
static unsigned _dma_dpcr;
union SpuUnion* _spu_RXX = (union SpuUnion*)&spu_RXX;
unsigned* dma_dpcr = &_dma_dpcr; // TODO this must be removed
volatile u16 _spu_RQ[10] = {0};

void SpuSetVoiceAttr(SpuVoiceAttr* arg) { NOT_IMPLEMENTED; }

long SpuSetReverbModeParam(SpuReverbAttr* attr) {
    extern s8 _spu_rev_param[];
    extern s32 _spu_rev_startaddr[];
    static SpuReverbAttr current;
    u_long mask;
    int mode_changed = 0;

    if (!attr)
        return -1;
    mask = attr->mask ? attr->mask : 0x1f;
    if (mask & 1) {
        int mode = attr->mode & ~0x100;
        const struct rev_param_entry* preset;
        if (mode < 0 || mode >= 10)
            return -1;
        preset = (const struct rev_param_entry*)(
            _spu_rev_param + mode * sizeof(struct rev_param_entry));
        for (unsigned i = 0; i < 32; i++) {
            const u16* values = &preset->dAPF1;
            Psyz_SpuWrite(offsetof(SPU_RXX, dAPF1) + i * 2, values[i]);
        }
        Psyz_SpuWrite(offsetof(SPU_RXX, rev_work_addr),
                      (u16)_spu_rev_startaddr[mode]);
        current.mode = mode;
        current.delay = mode == 7 || mode == 8 ? 0x7f : 0;
        current.feedback = mode == 7 ? 0x7f : 0;
        mode_changed = 1;
    }
    if (mask & 2) {
        current.depth.left = attr->depth.left;
        Psyz_SpuWrite(offsetof(SPU_RXX, rev_vol.left),
                      (u16)current.depth.left);
    } else if (mode_changed) {
        current.depth.left = 0;
        Psyz_SpuWrite(offsetof(SPU_RXX, rev_vol.left), 0);
    }
    if (mask & 4) {
        current.depth.right = attr->depth.right;
        Psyz_SpuWrite(offsetof(SPU_RXX, rev_vol.right),
                      (u16)current.depth.right);
    } else if (mode_changed) {
        current.depth.right = 0;
        Psyz_SpuWrite(offsetof(SPU_RXX, rev_vol.right), 0);
    }
    /* Rage Racer uses the fixed room/studio presets. Preserve the public
     * values for echo/delay callers; their variable tap rewrite can be added
     * when a title actually exercises modes 7 or 8. */
    if (mask & 8)
        current.delay = attr->delay;
    if (mask & 0x10)
        current.feedback = attr->feedback;
    return 0;
}

u_long _SpuSetAnyVoice(long on_off, u_long voice_bit, int arg2, int arg3) {
    NOT_IMPLEMENTED;
    return 0;
}

void SpuNGetVoiceAttr(int vNum, SpuVoiceAttr* arg) { NOT_IMPLEMENTED; }

long SpuGetKeyStatus(u_long voice_bit) {
    for (int voice = 0; voice < 24; voice++) {
        if ((voice_bit & ((u_long)1 << voice)) != 0)
            return Psyz_SpuVoiceKeyStatus(voice);
    }
    return -1;
}
