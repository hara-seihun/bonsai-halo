#pragma once

#include "../kernels/halo_kernels.h"

namespace halo {

// Re-score the saved DFlash2 block's candidates from begin_pos through row 7.
// predecessor is the corrected token immediately before begin_pos. The resulting
// DF_BLOCK - begin_pos tokens occupy draft_out[0..), not their original row offsets.
// The caller must keep topi, topv and hproj_out from the same completed block alive.
void launch_speculative_recycle(const DflashParams & block, int begin_pos, int predecessor, hipStream_t stream);

} // namespace halo
