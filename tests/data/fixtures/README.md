# Parser fixtures

Vendored test inputs (CC0 PDB archive data plus small hand-written files).

| File | Source | Notes |
|---|---|---|
| `1CRN`, `1UBQ`, `3FX2`, `2PTN` `.pdb.gz` / `.cif.gz` | RCSB PDB, downloaded 2026-11-12 | Same entries in both formats (cross-format tests) |
| `1L2Y.{pdb,cif}.gz` | RCSB PDB | NMR ensemble, 38 models, explicit hydrogens |
| `1EJG.{pdb,cif}.gz` | RCSB PDB | Crambin at 0.54 Å: 381 atoms with alternate locations (A/B/C) |
| `alanine.pqr` | hand-written | PDB2PQR-style whitespace PQR |
| `water.xyz` | hand-written | plain XYZ |
| `methane.extxyz` | hand-written | extended XYZ with a `radius` column |

Independent reference counts used by the tests: RCSB `deposited_atom_count`
(non-hydrogen atoms of the first model) and counts derived with `awk` from
the PDB files (see `crates/sesmesh/tests/structures.rs`).
