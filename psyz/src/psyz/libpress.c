#include <psyz.h>
#include <libpress.h>
#include <psyz/log.h>
#include <math.h>
#include <string.h>

/* Software MDEC.
 *
 * The PlayStation split movie decoding in two: the CPU turned the Huffman
 * bitstream into run-level codes (DecDCTvlc), and the MDEC chip turned those
 * into pixels (DecDCTin/DecDCTout). Both halves live here.
 *
 * Where the hardware was asynchronous and interrupt-driven, this decodes
 * eagerly and runs the completion callback before returning: callers that
 * poll with DecDCTinSync()/DecDCToutSync() always observe a finished
 * transfer, and callers driven by callbacks see them in the same order.
 */

#define MDEC_EOB 0xFE00
#define MDEC_MAX_CODES 0x20000
#define MDEC_MAX_PIXELS (640 * 480)

/* ------------------------------------------------------------------ */
/* Bit reader: halfwords are little-endian, bits consumed MSB first.   */
/* ------------------------------------------------------------------ */

typedef struct {
    const u_char* data;
    int size;
    int pos;
    unsigned long long acc;
    int count;
} BsReader;

static void BsInit(BsReader* reader, const u_char* data, int size) {
    reader->data = data;
    reader->size = size;
    reader->pos = 0;
    reader->acc = 0;
    reader->count = 0;
}

static void BsFill(BsReader* reader) {
    while (reader->count <= 48) {
        u_long half = 0;
        if (reader->pos + 1 < reader->size)
            half = (u_long)reader->data[reader->pos] |
                   ((u_long)reader->data[reader->pos + 1] << 8);
        reader->pos += 2;
        reader->acc |= (unsigned long long)half << (48 - reader->count);
        reader->count += 16;
    }
}

static u_long BsRead(BsReader* reader, int bits) {
    u_long value;
    BsFill(reader);
    value = (u_long)(reader->acc >> (64 - bits));
    reader->acc <<= bits;
    reader->count -= bits;
    return value;
}

/* ------------------------------------------------------------------ */
/* AC Huffman tables (BS v1/v2/v3), as 16-bit MDEC run-level values.   */
/* ------------------------------------------------------------------ */

static const u_short k_ac_zero1_low[2] = {0x0002, 0x0801};
static const u_short k_ac_zero2_high[2] = {0x1001, 0x0C01};
static const u_short k_ac_zero2_low[8] = {0x3401, 0x0006, 0x3001, 0x2C01,
                                          0x0C02, 0x0403, 0x0005, 0x2801};
static const u_short k_ac_zero3[4] = {0x1C01, 0x1801, 0x0402, 0x1401};
static const u_short k_ac_zero4[4] = {0x0802, 0x2401, 0x0004, 0x2001};
static const u_short k_ac_zero6[8] = {0x4001, 0x1402, 0x0007, 0x0803,
                                      0x0404, 0x3C01, 0x3801, 0x1002};
static const u_short k_ac_zero7[16] = {
    0x000B, 0x2002, 0x1003, 0x000A, 0x0804, 0x1C02, 0x5401, 0x5001,
    0x0009, 0x4C01, 0x4801, 0x0405, 0x0C03, 0x0008, 0x1802, 0x4401};
static const u_short k_ac_zero8[16] = {
    0x2802, 0x2402, 0x1403, 0x0C04, 0x0805, 0x0407, 0x0406, 0x000F,
    0x000E, 0x000D, 0x000C, 0x6801, 0x6401, 0x6001, 0x5C01, 0x5801};
static const u_short k_ac_zero9[16] = {
    0x001F, 0x001E, 0x001D, 0x001C, 0x001B, 0x001A, 0x0019, 0x0018,
    0x0017, 0x0016, 0x0015, 0x0014, 0x0013, 0x0012, 0x0011, 0x0010};
static const u_short k_ac_zero10[16] = {
    0x0028, 0x0027, 0x0026, 0x0025, 0x0024, 0x0023, 0x0022, 0x0021,
    0x0020, 0x040E, 0x040D, 0x040C, 0x040B, 0x040A, 0x0409, 0x0408};
static const u_short k_ac_zero11[16] = {
    0x0412, 0x0411, 0x0410, 0x040F, 0x1803, 0x4002, 0x3C02, 0x3802,
    0x3402, 0x3002, 0x2C02, 0x7C01, 0x7801, 0x7401, 0x7001, 0x6C01};

/* The trailing "s" bit negates the level held in the low 10 bits. */
static u_short AcValue(u_short base, u_long negate) {
    u_long run;
    long level;
    if (!negate)
        return base;
    run = (u_long)base >> 10;
    level = (long)(base & 0x3FF);
    return (u_short)((run << 10) | ((u_long)(-level) & 0x3FF));
}

/* Returns 0 for a coefficient, 1 for end-of-block, -1 on an invalid code. */
static int BsDecodeAc(BsReader* reader, u_short* out) {
    int zeros = 0;
    while (zeros < 12 && BsRead(reader, 1) == 0)
        zeros++;
    if (zeros >= 12)
        return -1;
    switch (zeros) {
    case 0:
        if (BsRead(reader, 1) == 0) {
            *out = MDEC_EOB;
            return 1;
        }
        *out = AcValue(0x0001, BsRead(reader, 1));
        return 0;
    case 1:
        if (BsRead(reader, 1)) {
            *out = AcValue(0x0401, BsRead(reader, 1));
            return 0;
        }
        *out = AcValue(k_ac_zero1_low[BsRead(reader, 1)], BsRead(reader, 1));
        return 0;
    case 2:
        if (BsRead(reader, 1)) {
            *out = AcValue(k_ac_zero2_high[BsRead(reader, 1)],
                           BsRead(reader, 1));
            return 0;
        }
        if (BsRead(reader, 1)) {
            *out = AcValue(0x0003, BsRead(reader, 1));
            return 0;
        }
        *out = AcValue(k_ac_zero2_low[BsRead(reader, 3)], BsRead(reader, 1));
        return 0;
    case 3:
        *out = AcValue(k_ac_zero3[BsRead(reader, 2)], BsRead(reader, 1));
        return 0;
    case 4:
        *out = AcValue(k_ac_zero4[BsRead(reader, 2)], BsRead(reader, 1));
        return 0;
    case 5:
        *out = (u_short)BsRead(reader, 16); /* escape: raw MDEC value */
        return 0;
    case 6:
        *out = AcValue(k_ac_zero6[BsRead(reader, 3)], BsRead(reader, 1));
        return 0;
    case 7:
        *out = AcValue(k_ac_zero7[BsRead(reader, 4)], BsRead(reader, 1));
        return 0;
    case 8:
        *out = AcValue(k_ac_zero8[BsRead(reader, 4)], BsRead(reader, 1));
        return 0;
    case 9:
        *out = AcValue(k_ac_zero9[BsRead(reader, 4)], BsRead(reader, 1));
        return 0;
    case 10:
        *out = AcValue(k_ac_zero10[BsRead(reader, 4)], BsRead(reader, 1));
        return 0;
    default:
        *out = AcValue(k_ac_zero11[BsRead(reader, 4)], BsRead(reader, 1));
        return 0;
    }
}

/* ------------------------------------------------------------------ */
/* Coefficient order and quantisation.                                 */
/* ------------------------------------------------------------------ */

static const u_char k_zigzag[64] = {
    0,  1,  5,  6,  14, 15, 27, 28, 2,  4,  7,  13, 16, 26, 29, 42,
    3,  8,  12, 17, 25, 30, 41, 43, 9,  11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63};

/* Standard PSX intra quantisation matrix, raster order. Overwritten by
 * DecDCTPutEnv() when the title supplies its own tables. */
static const u_char k_quant_raster[64] = {
    2,  16, 19, 22, 26, 27, 29, 34, 16, 16, 22, 24, 27, 29, 34, 37,
    19, 22, 26, 27, 29, 34, 34, 38, 22, 22, 26, 27, 29, 34, 37, 40,
    22, 26, 27, 29, 32, 35, 40, 48, 26, 27, 29, 32, 35, 40, 48, 58,
    26, 27, 29, 34, 38, 46, 56, 69, 27, 29, 35, 38, 46, 56, 69, 83};

/* MSVC only declares M_PI when _USE_MATH_DEFINES is set before <math.h>. */
#define MDEC_PI 3.14159265358979323846

static u_char s_zagzig[64]; /* stream index -> raster position */
static u_char s_quant_y[64];
static u_char s_quant_c[64];
static double s_cosine[8][8];
static int s_tables_ready;

static void MdecInitTables(void) {
    int index;
    int x;
    int u;
    if (s_tables_ready)
        return;
    for (index = 0; index < 64; index++)
        s_zagzig[k_zigzag[index]] = (u_char)index;
    for (index = 0; index < 64; index++) {
        s_quant_y[index] = k_quant_raster[s_zagzig[index]];
        s_quant_c[index] = k_quant_raster[s_zagzig[index]];
    }
    for (x = 0; x < 8; x++)
        for (u = 0; u < 8; u++)
            s_cosine[x][u] = (u == 0 ? sqrt(0.125) : 0.5) *
                             cos((2.0 * x + 1.0) * u * MDEC_PI / 16.0);
    s_tables_ready = 1;
}

/* ------------------------------------------------------------------ */
/* Decoder state.                                                      */
/* ------------------------------------------------------------------ */

static DecDCCb s_in_callback;
static DecDCCb s_out_callback;
static int s_vlc_limit;
static u_char s_pixels[MDEC_MAX_PIXELS * 3];
static int s_pixel_words;   /* decoded payload, in 32-bit words */
static int s_pixel_taken;   /* words already handed to DecDCTout() */
static int s_depth24;

static long SignedTenBit(u_long value) {
    return (long)(value & 0x3FF) - (long)((value & 0x200) << 1);
}

static long Clamp(long value, long low, long high) {
    return value < low ? low : (value > high ? high : value);
}

static void Idct(double* block) {
    double temp[64];
    double* src = block;
    double* dst = temp;
    int pass;
    for (pass = 0; pass < 2; pass++) {
        int row;
        for (row = 0; row < 8; row++) {
            int x;
            for (x = 0; x < 8; x++) {
                double sum = 0.0;
                int u;
                for (u = 0; u < 8; u++)
                    sum += s_cosine[x][u] * src[row * 8 + u];
                dst[x * 8 + row] = sum;
            }
        }
        {
            double* swap = src;
            src = dst;
            dst = swap;
        }
    }
    if (src != block)
        memcpy(block, src, sizeof(temp));
}

/* Decodes one 8x8 block. Returns codes consumed, or -1 when truncated. */
static int MdecDecodeBlock(const u_short* src, int available, double* block,
                           const u_char* quant) {
    short raw[64];
    int used = 0;
    int index = 0;
    u_long scale;
    long value;
    memset(raw, 0, sizeof(raw));
    while (used < available && src[used] == MDEC_EOB)
        used++; /* padding halfwords ahead of a block are ignored */
    if (used >= available)
        return -1;
    scale = ((u_long)src[used] >> 10) & 0x3F;
    value = SignedTenBit(src[used]) * (long)quant[0];
    used++;
    for (;;) {
        u_short code;
        if (scale == 0)
            value = SignedTenBit(src[used - 1]) * 2;
        raw[scale > 0 ? s_zagzig[index] : index] =
            (short)Clamp(value, -0x400, 0x3FF);
        if (used >= available)
            return -1;
        code = src[used++];
        index += (int)(((u_long)code >> 10) & 0x3F) + 1;
        if (index > 63)
            break;
        value = (SignedTenBit(code) * (long)quant[index] * (long)scale + 4) / 8;
    }
    for (index = 0; index < 64; index++)
        block[index] = raw[index];
    Idct(block);
    return used;
}

static long ToPixel(double value) {
    return Clamp((long)lrint(value), -128, 127) + 128;
}

/* ------------------------------------------------------------------ */
/* Public API.                                                         */
/* ------------------------------------------------------------------ */

void DecDCTReset(int mode) {
    (void)mode;
    MdecInitTables();
    s_pixel_words = 0;
    s_pixel_taken = 0;
}

int DecDCTBufSize(u_long* bs) {
    /* First halfword of a BS frame holds the run-level size in words. */
    return bs != NULL ? (int)(*(u_short*)bs) : 0;
}

DECDCTENV* DecDCTPutEnv(DECDCTENV* env) {
    int index;
    MdecInitTables();
    if (env == NULL)
        return NULL;
    for (index = 0; index < 64; index++) {
        s_quant_y[index] = env->iq_y[index];
        s_quant_c[index] = env->iq_c[index];
    }
    return env;
}

DECDCTENV* DecDCTGetEnv(DECDCTENV* env) {
    int index;
    MdecInitTables();
    if (env == NULL)
        return NULL;
    for (index = 0; index < 64; index++) {
        env->iq_y[index] = s_quant_y[index];
        env->iq_c[index] = s_quant_c[index];
    }
    return env;
}

int DecDCTvlcSize(int size) {
    int previous = s_vlc_limit;
    s_vlc_limit = size;
    return previous;
}

/* Huffman bitstream -> run-level codes, the software half of the pipeline.
 *
 * bs points at a BS frame header; buf receives that header's first two
 * halfwords followed by the run-level stream, which is what DecDCTin()
 * consumes. */
int DecDCTvlc(u_long* bs, u_long* buf) {
    BsReader reader;
    const u_char* bytes = (const u_char*)bs;
    u_short* out;
    u_short quant;
    int count;
    int capacity;

    MdecInitTables();
    if (bs == NULL || buf == NULL) {
        /* Resuming a suspended decode is not needed: this decoder never
         * stops early, so there is never anything to continue. */
        return 0;
    }

    out = (u_short*)buf;
    quant = (u_short)(bytes[4] | (bytes[5] << 8));
    out[0] = (u_short)(bytes[0] | (bytes[1] << 8));
    out[1] = (u_short)(bytes[2] | (bytes[3] << 8));

    capacity = s_vlc_limit > 0 ? s_vlc_limit * 2 : MDEC_MAX_CODES;
    BsInit(&reader, bytes + 8, out[0] * 4);
    count = 0;
    for (;;) {
        int block;
        for (block = 0; block < 6; block++) {
            u_long dc = BsRead(&reader, 10);
            if (block == 0 && dc == 0x1FF) {
                /* End of frame: pad to a whole word for the DMA-sized
                 * transfer the caller is about to issue. */
                if (count & 1)
                    out[2 + count++] = MDEC_EOB;
                return 0;
            }
            if (count + 8 >= capacity)
                return -1;
            out[2 + count++] = (u_short)((quant << 10) | (dc & 0x3FF));
            for (;;) {
                u_short code;
                int status = BsDecodeAc(&reader, &code);
                if (status < 0)
                    return -1;
                if (count >= capacity)
                    return -1;
                out[2 + count++] = code;
                if (status == 1)
                    break;
            }
        }
    }
}

int DecDCTvlc2(u_long* bs, u_long* buf, DECDCTTAB table) {
    (void)table;
    return DecDCTvlc(bs, buf);
}

void DecDCTvlcBuild(u_short* table) {
    /* The table is built into this decoder; nothing to expand. */
    (void)table;
}

int DecDCTvlcSize2(int size) { return DecDCTvlcSize(size); }

/* Run-level codes -> pixels.
 *
 * mode bit 0 selects 24-bit output, bit 1 sets the STP bit in 16-bit mode.
 * The frame geometry is not carried by the API, so it is recovered from the
 * code stream: every macroblock is six blocks, and the caller pulls the
 * result out in macroblock units through DecDCTout(). */
void DecDCTin(u_long* runlevel, int mode) {
    const u_short* codes;
    int available;
    int used = 0;
    u_char* dst = s_pixels;
    int limit;

    MdecInitTables();
    if (runlevel == NULL)
        return;

    s_depth24 = (mode & 1) != 0;
    s_pixel_words = 0;
    s_pixel_taken = 0;

    codes = (const u_short*)runlevel;
    available = (int)(codes[0]) * 2; /* header holds the word count */
    if (available <= 0 || available > MDEC_MAX_CODES)
        return;
    codes += 2; /* skip the two header halfwords */
    limit = MDEC_MAX_PIXELS / 256;

    while (used < available && limit-- > 0) {
        double blocks[6][64];
        int block;
        int x;
        int y;
        int ok = 1;
        for (block = 0; block < 6; block++) {
            const u_char* quant = block < 2 ? s_quant_c : s_quant_y;
            int step = MdecDecodeBlock(codes + used, available - used,
                                       blocks[block], quant);
            if (step < 0) {
                ok = 0;
                break;
            }
            used += step;
        }
        if (!ok)
            break;
        /* One macroblock: 16x16 pixels, written in raster order. */
        for (y = 0; y < 16; y++) {
            for (x = 0; x < 16; x++) {
                double cr = blocks[0][(x / 2) + (y / 2) * 8];
                double cb = blocks[1][(x / 2) + (y / 2) * 8];
                double luma =
                    blocks[2 + (y / 8) * 2 + (x / 8)][(x % 8) + (y % 8) * 8];
                long r = ToPixel(luma + 1.402 * cr);
                long g = ToPixel(luma - 0.3437 * cb - 0.7143 * cr);
                long b = ToPixel(luma + 1.772 * cb);
                if (s_depth24) {
                    *dst++ = (u_char)r;
                    *dst++ = (u_char)g;
                    *dst++ = (u_char)b;
                } else {
                    u_short packed = (u_short)(((r >> 3) & 0x1F) |
                                               (((g >> 3) & 0x1F) << 5) |
                                               (((b >> 3) & 0x1F) << 10));
                    if (mode & 2)
                        packed |= 0x8000;
                    *dst++ = (u_char)(packed & 0xFF);
                    *dst++ = (u_char)(packed >> 8);
                }
            }
        }
    }

    s_pixel_words = (int)(dst - s_pixels) / 4;
    if (s_in_callback != NULL)
        s_in_callback();
}

void DecDCTout(u_long* cell, int size) {
    int available;
    if (cell == NULL || size <= 0)
        return;
    available = s_pixel_words - s_pixel_taken;
    if (available < 0)
        available = 0;
    if (size <= available) {
        memcpy(cell, s_pixels + (size_t)s_pixel_taken * 4, (size_t)size * 4);
        s_pixel_taken += size;
    } else {
        if (available > 0)
            memcpy(cell, s_pixels + (size_t)s_pixel_taken * 4,
                   (size_t)available * 4);
        memset((u_char*)cell + (size_t)available * 4, 0,
               (size_t)(size - available) * 4);
        s_pixel_taken = s_pixel_words;
    }
    if (s_out_callback != NULL)
        s_out_callback();
}

DecDCCb DecDCTinCallback(DecDCCb func) {
    DecDCCb previous = s_in_callback;
    s_in_callback = func;
    return previous;
}

DecDCCb DecDCToutCallback(DecDCCb func) {
    DecDCCb previous = s_out_callback;
    s_out_callback = func;
    return previous;
}

int DecDCTinSync(int mode) {
    (void)mode;
    return 0; /* decoding already finished when DecDCTin() returned */
}

int DecDCToutSync(int mode) {
    (void)mode;
    return 0;
}
