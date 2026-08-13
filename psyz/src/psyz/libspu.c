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
    NOT_IMPLEMENTED;
    return 0;
}

u_long _SpuSetAnyVoice(long on_off, u_long voice_bit, int arg2, int arg3) {
    NOT_IMPLEMENTED;
    return 0;
}

void SpuNGetVoiceAttr(int vNum, SpuVoiceAttr* arg) { NOT_IMPLEMENTED; }

long SpuGetKeyStatus(u_long voice_bit) {
    for (int voice = 0; voice < 24; voice++) {
        if ((voice_bit & ((u_long)1 << voice)) != 0 &&
            Psyz_SpuRead(offsetof(SPU_RXX, voice[voice].volumex)) != 0)
            return SPU_ON;
    }
    return SPU_OFF;
}
