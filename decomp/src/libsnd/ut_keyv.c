#include "libsnd_private.h"
#ifdef __psyz
#include <stdio.h>
#include <stdlib.h>
#endif

short SsUtKeyOnV(short voice, short vabId, short prog, short tone, short note,
                 short fine, short voll, short volr) {
    int tn;

    if (_snd_ev_flag == 1) {
#ifdef __psyz
        if (getenv("PSYZ_SND_KEY_TRACE"))
            fprintf(stderr, "SsUtKeyOnV busy voice=%d vab=%d prog=%d tone=%d\n",
                    voice, vabId, prog, tone);
#endif
        return -1;
    }
    _snd_ev_flag = 1;
    if (voice < 0 || voice >= NUM_VOICES) {
        _snd_ev_flag = 0;
        return -1;
    }
    if (_SsVmVSetUp(vabId, prog)) {
#ifdef __psyz
        if (getenv("PSYZ_SND_KEY_TRACE"))
            fprintf(stderr,
                    "SsUtKeyOnV setup-failed voice=%d vab=%d used=%d prog=%d max=%d\n",
                    voice, vabId,
                    vabId >= 0 && vabId < NUM_VAB ? _svm_vab_used[vabId] : -1,
                    prog, kMaxPrograms);
#endif
        _snd_ev_flag = 0;
        return -1;
    }
    _svm_cur.seq_sep_no = 0x21;
    _svm_cur.note = note;
    _svm_cur.fine = fine;
    _svm_cur.tone = tone;
    if (voll == volr) {
        _svm_cur.pan = 0x40;
        _svm_cur.volume = voll;
    } else if (volr < voll) {
        _svm_cur.volume = voll;
        _svm_cur.pan = (volr << 6) / voll;
    } else {
        _svm_cur.volume = volr;
        _svm_cur.pan = 127 - (voll << 6) / volr;
    }
    _svm_cur.mvol = _svm_pg[prog].mvol;
    _svm_cur.mpan = _svm_pg[prog].mpan;
    _svm_cur.prog_tones = _svm_pg[prog].tones;
    tn = _svm_cur.tone + (_svm_cur.field_7_fake_program * 16);
    _svm_cur.tone_prior = _svm_tn[tn].prior;
    _svm_cur.tone_vag_idx = _svm_tn[tn].vag;
    _svm_cur.tone_vol = _svm_tn[tn].vol;
    _svm_cur.tone_pan = _svm_tn[tn].pan;
    _svm_cur.tone_center = _svm_tn[tn].center;
    _svm_cur.tone_shift = _svm_tn[tn].shift;
    _svm_cur.tone_mode = _svm_tn[tn].mode;
    _svm_cur.tone_min = _svm_tn[tn].min;
    _svm_cur.tone_max = _svm_tn[tn].max;
    if (_svm_cur.tone_vag_idx == 0) {
#ifdef __psyz
        if (getenv("PSYZ_SND_KEY_TRACE"))
            fprintf(stderr,
                    "SsUtKeyOnV empty-tone voice=%d vab=%d prog=%d actual=%d tone=%d\n",
                    voice, vabId, prog, _svm_cur.field_7_fake_program, tone);
#endif
        _snd_ev_flag = 0;
        return -1;
    }
    _svm_cur.voice = voice;
    _svm_voice[voice].unke = 0x21;
    _svm_voice[voice].vabId = vabId;
    _svm_voice[voice].unk10 = _svm_cur.field_7_fake_program;
    _svm_voice[voice].prog = prog;
    _svm_voice[voice].unk0 = _svm_cur.tone_vag_idx;
    _svm_voice[voice].tone = _svm_cur.tone;
    _svm_voice[voice].note = note;
    _svm_voice[voice].unk1b = 1;
    _svm_voice[voice].unk2 = 0;
    _SsVmDoAllocate();
    if (_svm_cur.tone_vag_idx == 0xFF) {
        vmNoiseOn(voice);
    } else {
        _SsVmKeyOnNow(1, note2pitch2(note, fine));
    }
#ifdef __psyz
    if (getenv("PSYZ_SND_KEY_TRACE"))
        fprintf(stderr,
                "SsUtKeyOnV key-on voice=%d vab=%d prog=%d actual=%d tone=%d vag=%d\n",
                voice, vabId, prog, _svm_cur.field_7_fake_program, tone,
                _svm_cur.tone_vag_idx);
#endif
    _snd_ev_flag = 0;
    return voice;
}

INCLUDE_ASM("asm/nonmatchings/libsnd/ut_keyv", SsUtKeyOffV);
