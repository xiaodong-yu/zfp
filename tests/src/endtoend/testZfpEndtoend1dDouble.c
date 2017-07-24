#include "src/encode1d.c"

#include "constants/1dDouble.h"
#include "utils/testMacros.h"
#include "utils/genSmoothRandNums.h"
#include "utils/hash64.c"
#include "zfpEndtoendBase.c"

int main()
{
  void* state;
  setupRandomData(&state);

  const struct CMUnitTest tests[] = {
    cmocka_unit_test_prestate(when_seededRandomSmoothDataGenerated_expect_ChecksumMatches, state),

    cmocka_unit_test_prestate_setup_teardown(given_1dDoubleArray_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec0, teardown, state),
    cmocka_unit_test_prestate_setup_teardown(given_1dDoubleArray_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec0, teardown, state),
    cmocka_unit_test_prestate_setup_teardown(given_1dDoubleArray_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec1, teardown, state),
    cmocka_unit_test_prestate_setup_teardown(given_1dDoubleArray_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec1, teardown, state),
    cmocka_unit_test_prestate_setup_teardown(given_1dDoubleArray_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec2, teardown, state),
    cmocka_unit_test_prestate_setup_teardown(given_1dDoubleArray_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec2, teardown, state),

    cmocka_unit_test_prestate_setup_teardown(given_1dDoubleArray_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate0, teardown, state),
    cmocka_unit_test_prestate_setup_teardown(given_1dDoubleArray_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate0, teardown, state),
    cmocka_unit_test_prestate_setup_teardown(given_1dDoubleArray_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate0, teardown, state),
    cmocka_unit_test_prestate_setup_teardown(given_1dDoubleArray_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate1, teardown, state),
    cmocka_unit_test_prestate_setup_teardown(given_1dDoubleArray_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate1, teardown, state),
    cmocka_unit_test_prestate_setup_teardown(given_1dDoubleArray_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate1, teardown, state),
    cmocka_unit_test_prestate_setup_teardown(given_1dDoubleArray_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate2, teardown, state),
    cmocka_unit_test_prestate_setup_teardown(given_1dDoubleArray_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate2, teardown, state),
    cmocka_unit_test_prestate_setup_teardown(given_1dDoubleArray_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate2, teardown, state),

    cmocka_unit_test_prestate_setup_teardown(given_1dDoubleArray_when_ZfpCompressFixedAccuracy_expect_BitstreamChecksumMatches, setupFixedAccuracy0, teardown, state),
    cmocka_unit_test_prestate_setup_teardown(given_1dDoubleArray_when_ZfpDecompressFixedAccuracy_expect_ArrayChecksumMatches, setupFixedAccuracy0, teardown, state),
    cmocka_unit_test_prestate_setup_teardown(given_1dDoubleArray_when_ZfpCompressFixedAccuracy_expect_CompressedValuesWithinAccuracy, setupFixedAccuracy0, teardown, state),
    cmocka_unit_test_prestate_setup_teardown(given_1dDoubleArray_when_ZfpCompressFixedAccuracy_expect_BitstreamChecksumMatches, setupFixedAccuracy1, teardown, state),
    cmocka_unit_test_prestate_setup_teardown(given_1dDoubleArray_when_ZfpDecompressFixedAccuracy_expect_ArrayChecksumMatches, setupFixedAccuracy1, teardown, state),
    cmocka_unit_test_prestate_setup_teardown(given_1dDoubleArray_when_ZfpCompressFixedAccuracy_expect_CompressedValuesWithinAccuracy, setupFixedAccuracy1, teardown, state),
    cmocka_unit_test_prestate_setup_teardown(given_1dDoubleArray_when_ZfpCompressFixedAccuracy_expect_BitstreamChecksumMatches, setupFixedAccuracy2, teardown, state),
    cmocka_unit_test_prestate_setup_teardown(given_1dDoubleArray_when_ZfpDecompressFixedAccuracy_expect_ArrayChecksumMatches, setupFixedAccuracy2, teardown, state),
    cmocka_unit_test_prestate_setup_teardown(given_1dDoubleArray_when_ZfpCompressFixedAccuracy_expect_CompressedValuesWithinAccuracy, setupFixedAccuracy2, teardown, state),
  };

  int result = cmocka_run_group_tests(tests, NULL, NULL);
  teardownRandomData(&state);

  return result;
}