# Copilot Instructions

## Project Guidelines
- Use naming convention in CUDA code: device buffers should use _d suffix and host reference buffers should use _h suffix.
- Use tolerance check for CPU vs GPU verification as `abs(cpu-gpu) <= max(aTol, rTol*max(abs(cpu), abs(gpu)))` with FP32 defaults `rTol=1e-5` and `aTol=1e-6`.