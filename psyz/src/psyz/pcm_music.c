#include <psyz/audio.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct PcmMusic {
    int16_t* samples;
    size_t frames;
    size_t cursor;
    int volume;
    int loop;
    int playing;
} PcmMusic;

static PcmMusic music = {.volume = 127};

static uint16_t read16le(const unsigned char* p) {
    return (uint16_t)(p[0] | (uint16_t)p[1] << 8);
}

static uint32_t read32le(const unsigned char* p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static short clamp16(int value) {
    if (value < -32768) return -32768;
    if (value > 32767) return 32767;
    return (short)value;
}

int Psyz_PcmMusicLoad(const char* path) {
    unsigned char header[44];
    int16_t* samples = NULL;
    size_t frames = 0;
    FILE* file = path ? fopen(path, "rb") : NULL;

    if (file && fread(header, 1, sizeof(header), file) == sizeof(header) &&
        read32le(header) == 0x46464952 && read32le(header + 8) == 0x45564157 &&
        read32le(header + 12) == 0x20746d66 && read32le(header + 16) == 16 &&
        read16le(header + 20) == 1 && read16le(header + 22) == 2 &&
        read32le(header + 24) == 44100 && read16le(header + 34) == 16 &&
        read32le(header + 36) == 0x61746164) {
        uint32_t bytes = read32le(header + 40);
        if (bytes > 0 && bytes % 4 == 0) {
            samples = malloc(bytes);
            if (samples && fread(samples, 1, bytes, file) == bytes)
                frames = bytes / 4;
        }
    }
    if (file) fclose(file);
    if (!frames) {
        free(samples);
        return -1;
    }

    Psyz_AudioLock();
    free(music.samples);
    music = (PcmMusic){samples, frames, 0, 127, 0, 0};
    Psyz_AudioUnlock();
    return 0;
}

void Psyz_PcmMusicUnload(void) {
    Psyz_AudioLock();
    free(music.samples);
    music = (PcmMusic){.volume = 127};
    Psyz_AudioUnlock();
}

void Psyz_PcmMusicPlay(int loop) {
    Psyz_AudioLock();
    if (music.samples) {
        music.cursor = 0;
        music.loop = loop != 0;
        music.playing = 1;
    }
    Psyz_AudioUnlock();
}

void Psyz_PcmMusicStop(void) {
    Psyz_AudioLock();
    music.playing = 0;
    music.cursor = 0;
    Psyz_AudioUnlock();
}

void Psyz_PcmMusicSetVolume(int volume) {
    if (volume < 0) volume = 0;
    if (volume > 127) volume = 127;
    Psyz_AudioLock();
    music.volume = volume;
    Psyz_AudioUnlock();
}

int Psyz_PcmMusicIsLoaded(void) {
    int loaded;
    Psyz_AudioLock();
    loaded = music.samples != NULL;
    Psyz_AudioUnlock();
    return loaded;
}

void Psyz_PcmMusicMix(short* out, int frames) {
    for (int frame = 0; frame < frames && music.playing; frame++) {
        size_t source = music.cursor * 2;
        out[frame * 2] = clamp16(out[frame * 2] +
                                 music.samples[source] * music.volume / 127);
        out[frame * 2 + 1] = clamp16(out[frame * 2 + 1] +
                                     music.samples[source + 1] * music.volume / 127);
        if (++music.cursor == music.frames) {
            music.cursor = 0;
            if (!music.loop) music.playing = 0;
        }
    }
}
