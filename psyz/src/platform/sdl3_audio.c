#include <psyz.h>
#include <psyz/log.h>
#include <SDL3/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <stdatomic.h>
#endif

#define N_CHANNELS 2
#define SAMPLE_SIZE sizeof(short)

static SDL_AudioStream* sdl_stream;
static SDL_Mutex* mutex;
static FILE* pcm_dump;
#ifdef _MSC_VER
static volatile __int64 rendered_frames;
static volatile __int64 rendered_energy;

static unsigned long long AudioCounterAdd(volatile __int64* counter,
                                          unsigned long long value) {
#ifdef __clang__
    return (unsigned long long)(_InterlockedExchangeAdd64(counter, (__int64)value) +
                                (__int64)value);
#else
    return (unsigned long long)_InterlockedAdd64(counter, (__int64)value);
#endif
}
static unsigned long long AudioCounterLoad(volatile __int64* counter) {
    return (unsigned long long)_InterlockedCompareExchange64(counter, 0, 0);
}
static void AudioCounterStore(volatile __int64* counter, unsigned long long value) {
    _InterlockedExchange64(counter, (__int64)value);
}
#else
static _Atomic unsigned long long rendered_frames;
static _Atomic unsigned long long rendered_energy;

static unsigned long long AudioCounterAdd(_Atomic unsigned long long* counter,
                                          unsigned long long value) {
    return atomic_fetch_add(counter, value) + value;
}
static unsigned long long AudioCounterLoad(_Atomic unsigned long long* counter) {
    return atomic_load(counter);
}
static void AudioCounterStore(_Atomic unsigned long long* counter,
                              unsigned long long value) {
    atomic_store(counter, value);
}
#endif

// Audio callback: SDL pulls audio from the SPU via this callback.
// The audio driver synchronizes the SPU based on the pulled samples.
static void SDLCALL audio_callback(void* userdata, SDL_AudioStream* stream,
                                   int additional_amount, int total_amount) {
    (void)userdata;
    (void)total_amount;
    if (additional_amount <= 0)
        return;

    int num_frames = additional_amount / (N_CHANNELS * SAMPLE_SIZE);
    short buf[4096] = {0};
    SDL_LockMutex(mutex);
    while (num_frames > 0) {
        int batch = num_frames;
        if (batch > 2048)
            batch = 2048;
        Psyz_SpuPullSamples(buf, batch);
        if (pcm_dump) {
            fwrite(buf, sizeof(short) * N_CHANNELS, (size_t)batch, pcm_dump);
            fflush(pcm_dump);
        }
        unsigned long long energy = 0;
        for (int i = 0; i < batch * N_CHANNELS; i++) {
            int sample = buf[i];
            energy += (unsigned long long)(sample < 0 ? -sample : sample);
        }
        AudioCounterAdd(&rendered_frames, (unsigned long long)batch);
        AudioCounterAdd(&rendered_energy, energy);
        SDL_PutAudioStreamData(stream, buf, batch * N_CHANNELS * SAMPLE_SIZE);
        num_frames -= batch;
    }
    SDL_UnlockMutex(mutex);
}

static bool is_audio_init = false;
int Psyz_AudioInit(void) {
    if (is_audio_init) {
        return 0;
    }
    Psyz_SpuInit();
    {
        const char* dump_path = getenv("PSYZ_AUDIO_PCM_DUMP");
        if (dump_path && *dump_path)
            pcm_dump = fopen(dump_path, "wb");
    }
    if (!SDL_WasInit(SDL_INIT_AUDIO)) {
        if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
            ERRORF("failed to init SDL audio: %s", SDL_GetError());
            return -1;
        }
    }

    SDL_AudioSpec spec = {
        .format = SDL_AUDIO_S16,
        .channels = N_CHANNELS,
        .freq = PSYZ_SPU_SAMPLE_RATE,
    };
    mutex = SDL_CreateMutex();
    if (!mutex) {
        ERRORF("failed to create audio mutex: %s", SDL_GetError());
        return -1;
    }

    sdl_stream = SDL_OpenAudioDeviceStream(
        SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, audio_callback, NULL);
    if (!sdl_stream) {
        ERRORF("failed to open audio device: %s", SDL_GetError());
        SDL_DestroyMutex(mutex);
        mutex = NULL;
        return -1;
    }
    /* Opening starts the stream paused.  The callback takes this mutex, so it
     * must exist before the device is resumed; previously the first callback
     * raced initialization and could drop or split the opening XA/VAB data. */
    SDL_ResumeAudioStreamDevice(sdl_stream);

    is_audio_init = true;
    DEBUGF("audio initialized");
    return 0;
}

void Psyz_AudioDestroy(void) {
    if (pcm_dump) {
        fclose(pcm_dump);
        pcm_dump = NULL;
    }
    if (mutex) {
        SDL_DestroyMutex(mutex);
        mutex = NULL;
    }
    if (sdl_stream) {
        SDL_DestroyAudioStream(sdl_stream);
        sdl_stream = NULL;
    }
    is_audio_init = false;
}

void Psyz_AudioLock() { SDL_LockMutex(mutex); }

void Psyz_AudioUnlock() { SDL_UnlockMutex(mutex); }

unsigned long long Psyz_AudioRenderedFrames(void) {
    return AudioCounterLoad(&rendered_frames);
}

unsigned long long Psyz_AudioRenderedEnergy(void) {
    return AudioCounterLoad(&rendered_energy);
}

void Psyz_AudioResetMetrics(void) {
    AudioCounterStore(&rendered_frames, 0);
    AudioCounterStore(&rendered_energy, 0);
}

void Psyz_AudioPause(void) { SDL_PauseAudioStreamDevice(sdl_stream); }

void Psyz_AudioUnpause(void) { SDL_ResumeAudioStreamDevice(sdl_stream); }
