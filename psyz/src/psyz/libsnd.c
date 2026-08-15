#include <psyz.h>
#include <libspu.h>
#include <libsnd.h>
#include <psyz/log.h>
#include "../../decomp/src/libspu/libspu_private.h"
#include "../../decomp/src/libsnd/libsnd_private.h"

#define LEN(x) ((s32)(sizeof(x) / sizeof(*(x))))
#define NUM_VOICES 24
#define PSYZ_SEQ_SLOTS 32
#define PSYZ_SEQ_CHANNELS 16

typedef struct PsyzSeqChannel {
    u8 program;
    u8 volume;
    u8 pan;
    u8 pitch_bend;
} PsyzSeqChannel;

typedef struct PsyzSeqState {
    const u8* start;
    const u8* cursor;
    u32 division;
    u32 tempo_us;
    u32 delta;
    u32 tick_accumulator;
    u8 running_status;
    u8 vab_id;
    u8 open;
    u8 playing;
    short loops;
    signed char voice_channel[NUM_VOICES];
    signed char voice_note[NUM_VOICES];
    u8 voice_velocity[NUM_VOICES];
    u32 voice_generation[NUM_VOICES];
    PsyzSeqChannel channels[PSYZ_SEQ_CHANNELS];
} PsyzSeqState;

static PsyzSeqState psyz_sequences[PSYZ_SEQ_SLOTS];
static unsigned long long psyz_seq_note_on_count;
static unsigned long long psyz_seq_voice_start_count;
static unsigned long long psyz_snd_pitch_update_count;
static u32 psyz_voice_generation[NUM_VOICES];

unsigned long long Psyz_SeqNoteOnCount(void) {
    return psyz_seq_note_on_count;
}

unsigned long long Psyz_SeqVoiceStartCount(void) {
    return psyz_seq_voice_start_count;
}

unsigned long long Psyz_SndPitchUpdateCount(void) {
    return psyz_snd_pitch_update_count;
}

static u32 read_variable_length(const u8** cursor) {
    u32 value = 0;
    u8 byte;
    do {
        byte = *(*cursor)++;
        value = (value << 7) | (byte & 0x7f);
    } while (byte & 0x80);
    return value;
}

static void reset_sequence_channels(PsyzSeqState* seq) {
    for (int channel = 0; channel < PSYZ_SEQ_CHANNELS; channel++) {
        seq->channels[channel].program = channel;
        seq->channels[channel].volume = 127;
        seq->channels[channel].pan = 64;
        seq->channels[channel].pitch_bend = 64;
    }
    for (int voice = 0; voice < NUM_VOICES; voice++) {
        seq->voice_channel[voice] = -1;
        seq->voice_note[voice] = -1;
        seq->voice_velocity[voice] = 0;
        seq->voice_generation[voice] = 0;
    }
}

static void sequence_key_off(PsyzSeqState* seq, int channel, int note) {
    for (short voice = 0; voice < NUM_VOICES; voice++) {
        if (seq->voice_channel[voice] == channel &&
            seq->voice_note[voice] == note) {
            int locked = _snd_ev_flag;
            _snd_ev_flag = 0;
            if (seq->voice_generation[voice] == psyz_voice_generation[voice])
                SsUtKeyOffV(voice);
            _snd_ev_flag = locked;
            seq->voice_channel[voice] = -1;
            seq->voice_note[voice] = -1;
            seq->voice_velocity[voice] = 0;
        }
    }
}

static void claim_sequence_voice(PsyzSeqState* owner, int voice, int channel,
                                 int note, int velocity) {
    /* Automatic allocation may steal a voice from another sequence/note. */
    for (int slot = 0; slot < PSYZ_SEQ_SLOTS; slot++) {
        psyz_sequences[slot].voice_channel[voice] = -1;
        psyz_sequences[slot].voice_note[voice] = -1;
        psyz_sequences[slot].voice_velocity[voice] = 0;
    }
    owner->voice_channel[voice] = channel;
    owner->voice_note[voice] = note;
    owner->voice_velocity[voice] = velocity;
    owner->voice_generation[voice] = psyz_voice_generation[voice];
}

static void update_sequence_pitch_bend(PsyzSeqState* seq, int channel) {
    PsyzSeqChannel* state = &seq->channels[channel];
    for (short voice = 0; voice < NUM_VOICES; voice++) {
        if (seq->voice_channel[voice] != channel)
            continue;
        SsUtPitchBend(voice, _svm_voice[voice].vabId,
                     _svm_voice[voice].prog, 0x3c, state->pitch_bend);
    }
}

static void sequence_key_on(
    PsyzSeqState* seq, int channel, int note, int velocity) {
    PsyzSeqChannel* state = &seq->channels[channel];
    int volume = velocity * state->volume / 127;
    int left = volume * _ss_score[seq - psyz_sequences][0].voll / 127;
    int right = volume * _ss_score[seq - psyz_sequences][0].volr / 127;
    short voice = -1;
    int tones;
    if (state->pan < 64)
        right = right * state->pan / 63;
    else if (state->pan > 64)
        left = left * (127 - state->pan) / 63;
    sequence_key_off(seq, channel, note);
    {
        int locked = _snd_ev_flag;
        _snd_ev_flag = 0;
        int setup = _SsVmVSetUp(seq->vab_id, state->program);
        if (setup == 0) {
            tones = _svm_pg[state->program].tones;
            if (getenv("PSYZ_SEQ_TRACE") && tones == 0)
                DEBUGF("SEQ program has no tones vab=%d program=%d actual=%d",
                       seq->vab_id, state->program,
                       _svm_cur.field_7_fake_program);
            for (int tone = 0; tone < tones; tone++) {
                int index = _svm_cur.field_7_fake_program * 16 + tone;
                if (getenv("PSYZ_SEQ_TRACE"))
                    DEBUGF("SEQ tone p=%d t=%d actual=%d range=%d..%d vag=%d",
                           state->program, tone, _svm_cur.field_7_fake_program,
                           _svm_tn[index].min, _svm_tn[index].max,
                           _svm_tn[index].vag);
                if (note < _svm_tn[index].min || note > _svm_tn[index].max)
                    continue;
                short allocated = SsUtKeyOn(
                    seq->vab_id, state->program, tone, note, 0, left, right);
                if (allocated >= 0) {
                    voice = allocated;
                    claim_sequence_voice(seq, allocated, channel, note, velocity);
                    psyz_seq_voice_start_count++;
                }
            }
        } else if (getenv("PSYZ_SEQ_TRACE")) {
            DEBUGF("SEQ VAB setup failed vab=%d program=%d used=%d max=%d",
                   seq->vab_id, state->program,
                   seq->vab_id < NUM_VAB ? _svm_vab_used[seq->vab_id] : -1,
                   kMaxPrograms);
        }
        _snd_ev_flag = locked;
    }
    if (voice >= 0)
        psyz_seq_note_on_count++;
    else if (getenv("PSYZ_SEQ_TRACE"))
        DEBUGF("SEQ note failed vab=%d channel=%d program=%d note=%d velocity=%d",
               seq->vab_id, channel, state->program, note, velocity);
}

static void stop_sequence_voices(PsyzSeqState* seq) {
    for (short voice = 0; voice < NUM_VOICES; voice++) {
        if (seq->voice_channel[voice] >= 0) {
            if (seq->voice_generation[voice] == psyz_voice_generation[voice])
                SsUtKeyOffV(voice);
            seq->voice_channel[voice] = -1;
            seq->voice_note[voice] = -1;
            seq->voice_velocity[voice] = 0;
        }
    }
}

static void restart_sequence(PsyzSeqState* seq) {
    seq->cursor = seq->start;
    seq->delta = read_variable_length(&seq->cursor);
    seq->running_status = 0;
    seq->tick_accumulator = 0;
    reset_sequence_channels(seq);
}

static int dispatch_sequence_event(PsyzSeqState* seq) {
    u8 status = *seq->cursor++;
    u8 data0;
    u8 data1;
    if (status < 0x80) {
        data0 = status;
        status = seq->running_status;
    } else {
        seq->running_status = status;
        data0 = *seq->cursor++;
    }
    switch (status & 0xf0) {
    case 0x80:
        data1 = *seq->cursor++;
        (void)data1;
        sequence_key_off(seq, status & 0xf, data0);
        break;
    case 0x90:
        data1 = *seq->cursor++;
        if (data1 == 0)
            sequence_key_off(seq, status & 0xf, data0);
        else
            sequence_key_on(seq, status & 0xf, data0, data1);
        break;
    case 0xb0:
        data1 = *seq->cursor++;
        if (data0 == 7) {
            seq->channels[status & 0xf].volume = data1;
        } else if (data0 == 10) {
            seq->channels[status & 0xf].pan = data1;
        }
        else if (data0 == 121) {
            seq->channels[status & 0xf].volume = 127;
            seq->channels[status & 0xf].pan = 64;
            seq->channels[status & 0xf].pitch_bend = 64;
            update_sequence_pitch_bend(seq, status & 0xf);
        }
        break;
    case 0xc0:
        seq->channels[status & 0xf].program = data0;
        break;
    case 0xd0:
        break;
    case 0xe0:
        data1 = *seq->cursor++;
        seq->channels[status & 0xf].pitch_bend = data1;
        update_sequence_pitch_bend(seq, status & 0xf);
        break;
    case 0xf0:
        if (status == 0xff) {
            u32 length = read_variable_length(&seq->cursor);
            if (data0 == 0x2f) {
                seq->cursor += length;
                return 0;
            }
            if (data0 == 0x51 && length == 3) {
                seq->tempo_us = ((u32)seq->cursor[0] << 16) |
                                ((u32)seq->cursor[1] << 8) | seq->cursor[2];
            }
            seq->cursor += length;
        } else {
            u32 length = data0;
            if (status == 0xf0 || status == 0xf7)
                length = read_variable_length(&seq->cursor);
            seq->cursor += length;
        }
        break;
    }
    seq->delta = read_variable_length(&seq->cursor);
    return 1;
}

typedef void (*SndSsMarkCallbackProc)(short seq_no, short sep_no, short data);

extern short _snd_seq_s_max;
extern short _snd_seq_t_max;
extern int _snd_ev_flag;
extern _SsFCALL SsFCALL;
extern SndSsMarkCallbackProc _SsMarkCallback[32][16];
extern struct SeqStruct* _ss_score[32];
extern unsigned int VBLANK_MINUS;
extern int _snd_openflag;

static void SetVoiceData(int nVoice, unsigned short* data) {
    for (int i = 0; i < 8; i++) {
        Psyz_SpuWrite(nVoice * 0x10 + i * 2, data[i]);
    }
}

static void SetStateData(unsigned short* data, unsigned nWords) {
    for (unsigned i = 0; i < nWords; i++) {
        Psyz_SpuWrite(0x180 + i * 2, data[i]);
    }
}

static unsigned short default_voice[] = {0, 0, 0x1000, 0x3000, 0x00BF, 0, 0, 0};
static unsigned short default_state[] = {
    0x3FFF, 0x3FFF, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000};
extern SPU_RXX* _svm_sreg;
void _SsVmInit(char num_voices);
void _SsInit(void) {
    int i, j;

    _svm_sreg = (SPU_RXX*)_spu_RXX;
    for (i = 0; i < NUM_VOICES; i++) {
        SetVoiceData(i, default_voice);
    }
    SetStateData(default_state, LEN(default_state));

    _SsVmInit(NUM_VOICES);
    for (j = 0; j < LEN(_SsMarkCallback); j++) {
        for (i = 0; i < LEN(*_SsMarkCallback); i++) {
            _SsMarkCallback[j][i] = NULL;
        }
    }

    VBLANK_MINUS = 60;
    _snd_openflag = 0;
    _snd_ev_flag = 0;
}

// void _SsVmFlush(void) { NOT_IMPLEMENTED; }

void _SsSeqPlay(short seq_no, short track_no) {
    PsyzSeqState* seq;
    (void)track_no;
    if (seq_no < 0 || seq_no >= PSYZ_SEQ_SLOTS)
        return;
    seq = &psyz_sequences[seq_no];
    if (!seq->open)
        return;
    if (getenv("PSYZ_SEQ_TRACE"))
        DEBUGF("SEQ tick slot=%d playing=%d delta=%u division=%u tempo=%u flags=%x",
               seq_no, seq->playing, seq->delta, seq->division, seq->tempo_us,
               _ss_score[seq_no][0].flags);
    if (!seq->playing) {
        seq->playing = 1;
        seq->loops = _ss_score[seq_no][0].unk20;
        restart_sequence(seq);
    }

    /* SsSeqCalledTbyT is invoked at the game's selected 60 Hz tick. Accumulate
     * MIDI ticks as a rational value so tempo changes do not drift. */
    seq->tick_accumulator += seq->division * 1000000u;
    while (seq->playing &&
           seq->tick_accumulator >= seq->tempo_us * VBLANK_MINUS) {
        seq->tick_accumulator -= seq->tempo_us * VBLANK_MINUS;
        if (seq->delta > 0)
            seq->delta--;
        while (seq->delta == 0) {
            if (dispatch_sequence_event(seq))
                continue;
            stop_sequence_voices(seq);
            if (seq->loops == 0 || --seq->loops > 0) {
                restart_sequence(seq);
            } else {
                seq->playing = 0;
                _ss_score[seq_no][0].flags &= ~SEQ_FLAG_1;
            }
            break;
        }
    }
}

void _SsSndTempo(short arg0, short arg1) { NOT_IMPLEMENTED; }

short SsSeqOpen(u_long* addr, short vab_id) {
    const u8* bytes = (const u8*)addr;
    int slot;
    PsyzSeqState* seq;
    /* PsyQ SEQ assets are commonly stored as the little-endian word 'SEQp',
     * i.e. bytes "pQES". Retail libsnd deliberately accepts either 'S' or
     * 'p' in byte zero and validates version byte 7. */
    if (!addr || (bytes[0] != 'S' && bytes[0] != 'p') || bytes[7] != 1) {
        ERRORF("SsSeqOpen got invalid SEQ header %02x %02x %02x %02x %02x %02x %02x %02x",
               bytes ? bytes[0] : 0, bytes ? bytes[1] : 0,
               bytes ? bytes[2] : 0, bytes ? bytes[3] : 0,
               bytes ? bytes[4] : 0, bytes ? bytes[5] : 0,
               bytes ? bytes[6] : 0, bytes ? bytes[7] : 0);
        return -1;
    }
    for (slot = 0; slot < _snd_seq_s_max && slot < PSYZ_SEQ_SLOTS; slot++)
        if (!(_snd_openflag & (1 << slot)))
            break;
    if (slot >= _snd_seq_s_max || slot >= PSYZ_SEQ_SLOTS)
        return -1;

    seq = &psyz_sequences[slot];
    memset(seq, 0, sizeof(*seq));
    seq->division = ((u32)bytes[8] << 8) | bytes[9];
    seq->tempo_us = ((u32)bytes[10] << 16) | ((u32)bytes[11] << 8) | bytes[12];
    if (seq->division == 0 || seq->tempo_us == 0) {
        ERRORF("SsSeqOpen got invalid timing data");
        return -1;
    }
    seq->vab_id = vab_id;
    seq->start = bytes + 15;
    seq->open = 1;
    restart_sequence(seq);
    if (getenv("PSYZ_SEQ_TRACE"))
        DEBUGF("SEQ open slot=%d division=%u tempo=%u delta=%u first=%02x",
               slot, seq->division, seq->tempo_us, seq->delta, *seq->cursor);
    _snd_openflag |= 1 << slot;
    return slot;
}

void SsSetMarkCallback(
    short access_num, short seq_num, SsMarkCallbackProc proc) {
    NOT_IMPLEMENTED;
}

void SsSeqSetDecrescendo(short seq_access_num, short vol, long v_time) {
    NOT_IMPLEMENTED;
}

void _SsVmGetSeqVol(short seq_sep_no, short* voll, short* volr) {
    NOT_IMPLEMENTED;
}

int _SsInitSoundSep(short flag, short i, short vab_id, unsigned long* addr) {
    NOT_IMPLEMENTED;
    return -1;
}

char _SsVmAlloc(short voice) {
    int selected = _SsVmMaxVoice;
    int candidate = -1;
    unsigned short best_pitch = 0xffff;
    unsigned short best_age = 0;
    unsigned short threshold = (unsigned char)_svm_cur.tone_prior;
    (void)voice;

    for (int i = 0; i < _SsVmMaxVoice; i++) {
        if (!_svm_voice[i].unk1b && !_svm_voice[i].unk6) {
            selected = i;
            break;
        }

        if ((unsigned short)_svm_voice[i].unk18 < threshold) {
            threshold = (unsigned short)_svm_voice[i].unk18;
            candidate = i;
            best_pitch = _svm_voice[i].unk6;
            best_age = (unsigned short)_svm_voice[i].unk2;
        } else if ((unsigned short)_svm_voice[i].unk18 == threshold &&
                   (_svm_voice[i].unk6 < best_pitch ||
                    (_svm_voice[i].unk6 == best_pitch &&
                     (unsigned short)_svm_voice[i].unk2 > best_age))) {
            candidate = i;
            best_pitch = _svm_voice[i].unk6;
            best_age = (unsigned short)_svm_voice[i].unk2;
        }
    }

    if (selected == _SsVmMaxVoice && candidate >= 0)
        selected = candidate;
    if (selected < _SsVmMaxVoice) {
        for (int i = 0; i < _SsVmMaxVoice; i++)
            _svm_voice[i].unk2++;
        _svm_voice[selected].unk2 = 0;
        _svm_voice[selected].unk18 = _svm_cur.tone_prior;
        psyz_voice_generation[selected]++;
    }
    return selected;
}

void _SsVmSetSeqVol(
    short seq_sep_no, unsigned short voll, unsigned short volr, short arg3) {
    int seq_no = seq_sep_no & 0xff;
    int track_no = (seq_sep_no >> 8) & 0xff;
    (void)arg3;
    if (seq_no >= 0 && seq_no < _snd_seq_s_max &&
        track_no >= 0 && track_no < _snd_seq_t_max) {
        _ss_score[seq_no][track_no].voll = voll > 127 ? 127 : voll;
        _ss_score[seq_no][track_no].volr = volr > 127 ? 127 : volr;
    }
}

void _SsVmSeqKeyOff(s16 seq_sep_num) {
    int seq_no = seq_sep_num & 0xff;
    if (seq_no < 0 || seq_no >= PSYZ_SEQ_SLOTS)
        return;
    stop_sequence_voices(&psyz_sequences[seq_no]);
    psyz_sequences[seq_no].playing = 0;
}

short SsUtKeyOffV(short voice) {
    if (voice < 0 || voice >= NUM_VOICES)
        return -1;
    _svm_cur.voice = voice;
    _SsVmKeyOffNow(0);
    _svm_voice[voice].auto_pan = 0;
    _svm_voice[voice].auto_vol = 0;
    return 0;
}

static short set_voice_pitch(short voice, short vabId, short prog,
                             short note, short fine) {
    if (voice < 0 || voice >= NUM_VOICES ||
        _svm_voice[voice].vabId != vabId ||
        _svm_voice[voice].prog != prog || _SsVmVSetUp(vabId, prog) != 0) {
        if (getenv("PSYZ_SND_PITCH_TRACE"))
            fprintf(stderr,
                    "set_voice_pitch reject voice=%d vab=%d(cur %d) "
                    "prog=%d(cur %d)\n",
                    voice, vabId,
                    voice >= 0 && voice < NUM_VOICES
                        ? _svm_voice[voice].vabId : -99,
                    prog,
                    voice >= 0 && voice < NUM_VOICES
                        ? _svm_voice[voice].prog : -99);
        return -1;
    }
    _svm_cur.tone = _svm_voice[voice].tone;
    _svm_cur.field_7_fake_program = _svm_voice[voice].unk10;
    _svm_sreg_buf[voice].pitch = note2pitch2(note, fine);
    _svm_sreg_dirty[voice] |= 4;
    /* PsyQ's SsUtChangePitch only writes the pitch register; the voice
     * keeps its key-on note. Rage Racer relies on that: it always passes
     * old_note 0x3C (the note it keyed with), so recording new_note here
     * made the very next update fail and froze every pitched effect
     * (engine revs, skids, impacts) after its first change. */
    _svm_voice[voice].unk04 = _svm_sreg_buf[voice].pitch;
    psyz_snd_pitch_update_count++;
    return 0;
}

short SsUtChangePitch(short voice, short vabId, short prog, short old_note,
                      short old_fine, short new_note, short new_fine) {
    (void)old_fine;
    if (voice < 0 || voice >= NUM_VOICES ||
        _svm_voice[voice].note != old_note) {
        if (getenv("PSYZ_SND_PITCH_TRACE"))
            fprintf(stderr,
                    "SsUtChangePitch reject voice=%d old_note=%d(cur %d)\n",
                    voice, old_note,
                    voice >= 0 && voice < NUM_VOICES
                        ? _svm_voice[voice].note : -99);
        return -1;
    }
    return set_voice_pitch(voice, vabId, prog, new_note, new_fine);
}

short SsUtPitchBend(
    short voice, short vabId, short prog, short note, short pbend) {
    int bend;
    int scaled;
    int base;
    int fine;
    int tone_index;
    (void)note;
    if (voice < 0 || voice >= NUM_VOICES ||
        _svm_voice[voice].unke != 0x21 ||
        _svm_voice[voice].vabId != vabId ||
        _svm_voice[voice].prog != prog || _SsVmVSetUp(vabId, prog) != 0)
        return -1;

    /* Match PsyQ's asymmetric bend calculation: 64 is centre, and each
     * side is scaled by the selected tone's VAB bend limit. */
    tone_index = _svm_voice[voice].unk10 * 16 + _svm_voice[voice].tone;
    bend = pbend - 64;
    base = _svm_voice[voice].note;
    fine = 0;
    if (bend > 0) {
        scaled = bend * _svm_tn[tone_index].pbmax;
        base += scaled / 63;
        fine = (scaled % 63) * 2;
    } else if (bend < 0) {
        scaled = bend * _svm_tn[tone_index].pbmin;
        /* Arithmetic division toward minus infinity, as in the MIPS code. */
        int quotient = scaled >= 0 ? scaled / 64 : -((-scaled + 63) / 64);
        base += quotient - 1;
        fine = (scaled - quotient * 64) * 2 + 127;
    }
    return set_voice_pitch(voice, vabId, prog, base, fine);
}

void vmNoiseOn(char voice) { NOT_IMPLEMENTED; }

void SetAutoVol(short voices) { NOT_IMPLEMENTED; }

void SeAutoPan(short arg0, short arg1, short arg2, short arg3) {
    NOT_IMPLEMENTED;
}

void _SsNoteOn(short a0, short a1, unsigned char a2, unsigned char a3) {
    NOT_IMPLEMENTED;
}
void _SsSetProgramChange(short a0, short a1, unsigned char a2) {
    NOT_IMPLEMENTED;
}
void _SsGetMetaEvent(short a0, short a1, unsigned char a2) { NOT_IMPLEMENTED; }
void _SsSetPitchBend(short a0, short a1) { NOT_IMPLEMENTED; }
void _SsSetControlChange(short a0, short a1, unsigned char a2) {
    NOT_IMPLEMENTED;
}
void _SsContBankChange(short a0, short a1) { NOT_IMPLEMENTED; }
void _SsContDataEntry(short a0, short a1, unsigned char a2) { NOT_IMPLEMENTED; }
void _SsContMainVol(short a0, short a1, unsigned char a2) { NOT_IMPLEMENTED; }
void _SsContPanpot(short a0, short a1, unsigned char a2) { NOT_IMPLEMENTED; }
void _SsContExpression(short a0, short a1, unsigned char a2) {
    NOT_IMPLEMENTED;
}
void _SsContDamper(short a0, short a1, unsigned char a2) { NOT_IMPLEMENTED; }
void _SsContExternal(short a0, short a1, unsigned char a2) { NOT_IMPLEMENTED; }
void _SsContNrpn1(short a0, short a1, unsigned char a2) { NOT_IMPLEMENTED; }
void _SsContNrpn2(short a0, short a1, unsigned char a2) { NOT_IMPLEMENTED; }
void _SsContRpn1(short a0, short a1, unsigned char a2) { NOT_IMPLEMENTED; }
void _SsContRpn2(short a0, short a1, unsigned char a2) { NOT_IMPLEMENTED; }
void _SsContResetAll(short a0, short a1) { NOT_IMPLEMENTED; }

void _SsSetNrpnVabAttr0(
    short a0, short a1, short a2, VagAtr a3, short a4, unsigned char a5) {
    NOT_IMPLEMENTED;
}
void _SsSetNrpnVabAttr1(
    short a0, short a1, short a2, VagAtr a3, short a4, unsigned char a5) {
    NOT_IMPLEMENTED;
}
void _SsSetNrpnVabAttr2(
    short a0, short a1, short a2, VagAtr a3, short a4, unsigned char a5) {
    NOT_IMPLEMENTED;
}
void _SsSetNrpnVabAttr3(
    short a0, short a1, short a2, VagAtr a3, short a4, unsigned char a5) {
    NOT_IMPLEMENTED;
}
void _SsSetNrpnVabAttr4(
    short a0, short a1, short a2, VagAtr a3, short a4, unsigned char a5) {
    NOT_IMPLEMENTED;
}
void _SsSetNrpnVabAttr5(
    short a0, short a1, short a2, VagAtr a3, short a4, unsigned char a5) {
    NOT_IMPLEMENTED;
}
void _SsSetNrpnVabAttr6(
    short a0, short a1, short a2, VagAtr a3, short a4, unsigned char a5) {
    NOT_IMPLEMENTED;
}
void _SsSetNrpnVabAttr7(
    short a0, short a1, short a2, VagAtr a3, short a4, unsigned char a5) {
    NOT_IMPLEMENTED;
}
void _SsSetNrpnVabAttr8(
    short a0, short a1, short a2, VagAtr a3, short a4, unsigned char a5) {
    NOT_IMPLEMENTED;
}
void _SsSetNrpnVabAttr9(
    short a0, short a1, short a2, VagAtr a3, short a4, unsigned char a5) {
    NOT_IMPLEMENTED;
}
void _SsSetNrpnVabAttr10(
    short a0, short a1, short a2, VagAtr a3, short a4, unsigned char a5) {
    NOT_IMPLEMENTED;
}
void _SsSetNrpnVabAttr11(
    short a0, short a1, short a2, VagAtr a3, short a4, unsigned char a5) {
    NOT_IMPLEMENTED;
}
void _SsSetNrpnVabAttr12(
    short a0, short a1, short a2, VagAtr a3, short a4, unsigned char a5) {
    NOT_IMPLEMENTED;
}
void _SsSetNrpnVabAttr13(
    short a0, short a1, short a2, VagAtr a3, short a4, unsigned char a5) {
    NOT_IMPLEMENTED;
}
void _SsSetNrpnVabAttr14(
    short a0, short a1, short a2, VagAtr a3, short a4, unsigned char a5) {
    NOT_IMPLEMENTED;
}
void _SsSetNrpnVabAttr15(
    short a0, short a1, short a2, VagAtr a3, short a4, unsigned char a5) {
    NOT_IMPLEMENTED;
}
void _SsSetNrpnVabAttr16(
    short a0, short a1, short a2, VagAtr a3, short a4, unsigned char a5) {
    NOT_IMPLEMENTED;
}
void _SsSetNrpnVabAttr17(
    short a0, short a1, short a2, VagAtr a3, short a4, unsigned char a5) {
    NOT_IMPLEMENTED;
}
void _SsSetNrpnVabAttr18(
    short a0, short a1, short a2, VagAtr a3, short a4, unsigned char a5) {
    NOT_IMPLEMENTED;
}
void _SsSetNrpnVabAttr19(
    short a0, short a1, short a2, VagAtr a3, short a4, unsigned char a5) {
    NOT_IMPLEMENTED;
}
