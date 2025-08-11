# Grammar parity and provenance

AssemblyMC approximates the assembly index by repeatedly splitting a molecule
into fragments.  Our diffusion grammar **G** grows molecules via bond edits and
atom insertions.  The table below maps the concepts:

| AssemblyMC step | Grammar production | Label |
|-----------------|-------------------|-------|
| Split fragment into two parts | Inverse of a bond edit in **G** | `G_MC`

Because a Monte-Carlo step is not identical to any single production in **G**, runs
that rely on the AssemblyMC estimator should be recorded with the grammar label
`G_MC`.  If future work proves the operations to be exactly equivalent,
`G` may be used instead.

## Provenance logging

When `ai.method=assemblymc`, the run logger captures additional
information for reproducibility:

- Commit hash of the local repository.
- Operating system version.
- Full command line used to invoke the program.

This metadata is emitted in the JSON header produced by
`assembly_diffusion.run_logger.init_run_logger`, enabling downstream
consumers to trace the exact environment used for AssemblyMC results.
