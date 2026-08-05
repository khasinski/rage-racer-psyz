#include <gtest/gtest.h>

extern "C" __attribute__((weak)) void target_setup(void) {}
extern "C" __attribute__((weak)) void target_skips(void) {}
extern "C" __attribute__((weak)) void target_report(void) {}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    target_setup();
    int rc = RUN_ALL_TESTS();
    target_report();
    return rc;
}
