#ifndef PSYZ_VSYNC_TIMING_H
#define PSYZ_VSYNC_TIMING_H

/* The VSync(1) root-counter value for a given elapsed time, in the
 * 0x100-per-VBlank units the BIOS counter uses.
 *
 * The counter and the presentation deadline may run at different rates: a
 * PAL game simulates at 50 Hz (a 0x180 two-field wait must take 40 ms)
 * while the host presents at its own refresh (the first field is ready at
 * the presentation deadline, so a single-field 0x80 wait keeps the host
 * cadence). Unit one therefore arrives with the first presentation and
 * every later unit on the counter-rate grid. */
static inline unsigned Psyz_VsyncCounterUnits(double elapsed_us,
                                              double present_period_us,
                                              double counter_period_us) {
    unsigned units;
    if (elapsed_us < present_period_us)
        return 0;
    units = (unsigned)(elapsed_us / counter_period_us);
    if (units < 1)
        units = 1;
    return units * 0x100u;
}

#endif
