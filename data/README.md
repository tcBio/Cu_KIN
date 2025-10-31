# Data Assets

Place packed genotype binaries here when running large-scale benchmarks. Files in this directory are ignored by git to keep the repository lightweight.

## Expected Layout
- Samples are stored contiguously: all loci for sample 0, then all loci for sample 1, etc.
- Each locus is encoded as two bits (`00`, `01`, `10` -> genotypes 0,1,2). The lower two bits of each byte are used; additional bits are ignored.
- Thirty-two loci are packed into a 64-bit little-endian word. Locus 0 occupies bits [1:0], locus 1 bits [3:2], and so on.

## Regenerating Mock Data
- Use `kinship::pack_genotypes_2bit` from `src/bit_encode.cpp` to convert byte-oriented genotypes into the packed representation.
- The sample app in `src/main.cu` demonstrates generating deterministic mock genotypes for quick smoke tests.

Document each binary with a short note (samples, loci, origin) so collaborators can reproduce the data source if needed.
