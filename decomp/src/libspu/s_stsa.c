#include "libspu_private.h"

unsigned SpuSetTransferStartAddr(unsigned addr) {
    unsigned _addr;
    _addr = addr;
    /* Sticky VAB banks may legally start at 0x1000. Rage Racer's primary
     * bank does exactly that. Rejecting it left _spu_tsa at zero, while the
     * VAB address table still included the 0x1000 base, so every voice read
     * sample data 0x1000 bytes after the bytes actually transferred. */
    if (_addr < 0x1000 || _addr > 0x7FFFF) {
        return 0;
    }
    _addr = _spu_FsetRXXa(-1, _addr);
    _spu_tsa = (u16)_addr;
    return (u16)_addr << _spu_mem_mode_plus;
}
