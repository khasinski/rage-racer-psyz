#ifndef PSYZ_AUDIO_H
#define PSYZ_AUDIO_H

/**
 * @file audio.h
 * @brief Host audio subsystem backend endpoints.
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the audio subsystem
 *
 * Opens the host audio device and the SPU mixer. Idempotent.
 *
 * @return 0 on success, -1 on failure
 */
int Psyz_AudioInit(void);

/**
 * @brief Shut down the audio subsystem and release its resources
 *
 * Idempotent, including after failed initialization. Stops the host callback
 * before releasing its dependencies. Call on the lifecycle/main thread, not
 * from an audio callback or while holding Psyz_AudioLock, and before SDL_Quit.
 * Does not reset disc/sequence/game state or shut down SDL's audio subsystem.
 */
void Psyz_AudioDestroy(void);

/**
 * @brief Pause the host audio device so it stops pulling samples from the SPU
 *
 * Psyz_AudioUnpause must be called to resume.
 */
void Psyz_AudioPause(void);

/**
 * @brief Resume audio playback after Psyz_AudioPause
 */
void Psyz_AudioUnpause(void);

/**
 * @brief Acquire the audio mutex
 *
 * Intended for tests and offline rendering that need to suspend the SDL audio
 * callback while pulling samples directly. Must be paired with
 * Psyz_AudioUnlock.
 */
void Psyz_AudioLock(void);

/**
 * @brief Release the audio mutex acquired with Psyz_AudioLock
 */
void Psyz_AudioUnlock(void);

/** Debug/test counters for PCM actually produced by the SPU mixer. */
unsigned long long Psyz_AudioRenderedFrames(void);
unsigned long long Psyz_AudioRenderedEnergy(void);
/** Number of MIDI note-on events dispatched by the host SEQ player. */
unsigned long long Psyz_SeqNoteOnCount(void);
/** Number of VAB tone voices started by SEQ notes (may exceed note count). */
unsigned long long Psyz_SeqVoiceStartCount(void);
/** Number of live voice-pitch changes issued through libsnd. */
unsigned long long Psyz_SndPitchUpdateCount(void);
void Psyz_AudioResetMetrics(void);

/** Load a 44100 Hz, stereo, signed 16-bit PCM WAV into the music stream. */
int Psyz_PcmMusicLoad(const char* path);
void Psyz_PcmMusicUnload(void);
void Psyz_PcmMusicPlay(int loop);
void Psyz_PcmMusicStop(void);
void Psyz_PcmMusicSetVolume(int volume);
int Psyz_PcmMusicIsLoaded(void);

/** Raw emulated SPU register base for native fixed-address game code. */
void *Psyz_SpuRegisterBase(void);

#ifdef __cplusplus
}
#endif

#endif
