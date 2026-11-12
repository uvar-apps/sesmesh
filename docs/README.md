# sesmesh documentation

Planning documents for **sesmesh**, a Rust library and CLI that computes the
Solvent Excluded Surface (Connolly surface) of molecules from PDB, PDBx/mmCIF
and XYZ files and writes triangle meshes with normals (OBJ).

| Document | Contents |
|---|---|
| [01 — Product Specification](01-product-specification.md) | Background and literature survey, goals/non-goals, use cases, functional and non-functional requirements (IDs `FR-*`, `NFR-*`), acceptance criteria, open questions, references `[R1]–[R54]`. |
| [02 — Design](02-design.md) | Mathematical foundation (exact implicit SES field), architecture, data model, input readers, geometry precomputation, sparse brick field evaluation with SIMD, Marching Cubes with exact edge refinement and analytic normals, post-processing, output formats, concurrency, API/CLI, verification, risks, alternatives. |
| [03 — Implementation Plan](03-implementation-plan.md) | Phased task breakdown (T0.1–T10.5) with tests and exit criteria, milestones v0.1–v1.0, dependencies, test-data and benchmark protocol, effort estimate. |

Status: draft for review — no code yet.
