# Porting the RSAFDQ2022 3D-0D operator onto FerriteOperators 0.4

Working document. Target: delete `src/discretization/rsafdq-operator.jl` and let the
3D-0D coupled operator be an ordinary `setup_operator` result.

Repos this was written against:

| | path | branch / version |
|---|---|---|
| Thunderbolt | `/home/dogiermann/Repos/Thunderbolt.jl` | `do/fo-v0.4`, `c5387636` |
| FerriteOperators | `/home/dogiermann/Repos/FerriteOperators.jl` | `do/rework-internal-engine`, v0.4.0 |
| Ferrite | `/home/dogiermann/Repos/Ferrite.jl` | `do/block-constraint-application` |

**Thunderbolt does not compile against either of the two branches today.**
`/home/dogiermann/Repos/Thunderbolt.jl/Project.toml:69` pins `FerriteOperators = "0.3.7"`
and `Manifest.toml:550-558` resolves FO 0.3.7 from the registry (no `path =`); `Ferrite`
is registry 1.6.0 (`Manifest.toml:526-536`). The branch name `do/fo-v0.4` is aspirational.
Everything below assumes the package-wide FO 0.3 → 0.4 migration lands first — see
[§4 Step 0](#step-0--prerequisite-the-package-wide-fo-04-migration).

---

## 1. Current state

### 1.1 The operator

`AssembledRSAFDQ2022Operator` — `src/discretization/rsafdq-operator.jl:2`, seven untyped
`@concrete` fields `J, strategy, subdomain_caches, dh, integrator, chambers, tying_caches`.
It is the **only** subtype of `FerriteOperators.AbstractBlockOperator` in either repo, and
the name never appears outside its own file. Its own file header already says
`# TODO try to reproduce this via the BlockOperator`.

Built at `rsafdq-operator.jl:146-189` by the single `setup_stage_operator` method
`(f::RSAFDQ20223DFunction, ::HomotopyPathSolver, local_solver_cache, t₀)`, reached from
`src/solver/time/homotopy.jl:183`. That method shadows the generic continuation method at
`homotopy.jl:98`, which is a one-liner: `setup_operator(get_strategy(f), get_volume_integrator(f), f.dh)`.
**The whole port is about getting back to that one-liner.**

### 1.2 The augmented facet dofs

The chamber pressures are **not** in the `DofHandler`. They are appended by arithmetic:

```julia
# src/modeling/rsafdq2022.jl:184
pressure_dof_index = num_unknowns_structure + i     # num_unknowns_structure = solution_size(structural_problem)
```

stored twice on the chamber (`pressure_dof_index_local` / `_global`,
`rsafdq2022.jl:4-5`, identical values today), and spliced into the local system per facet:

```julia
# src/discretization/rsafdq-operator.jl:27
dofs = [celldofs(facet); chamber.pressure_dof_index_local]
```

The element side mirrors the convention independently: `Pressure3D0DVolumeCouplerCache`
sets `pressure_dof = ndofs_per_cell(sdh)+1` at `src/modeling/coupler/fsi.jl:102`. Two
places agree by hand that the pressure sits last in the local system — that agreement is
exactly what `global_dof_range` makes a framework contract.

Consequence, documented in `_check_rsafdq_internal_variables`
(`rsafdq2022.jl:144-171`): the *solution vector* is `[u_dofs | internal_vars | pressures]`
while the *matrix* is `[u_dofs | pressures]` (`block_sizes = [ndofs(dh), num_chambers]`,
`rsafdq-operator.jl:169`). One `dofs` array indexes both, so materials with internal
variables are rejected outright.

### 1.3 The placeholder-`ones` block sparsity

```julia
# src/discretization/rsafdq-operator.jl:172-178
Jblock = BlockArray(spzeros(total_size, total_size), block_sizes, block_sizes)
Jblock[Block(1, 1)] = J
Jblock[Block(1, 2)] = sparse(ones(ndofs(dh), num_chambers))    # TODO optimize storage
Jblock[Block(2, 1)] = sparse(ones(num_chambers, ndofs(dh)))
Jblock[Block(2, 2)] = sparse(ones(num_chambers, num_chambers))
Ferrite.fillzero!(Jblock)
```

`sparse(ones(m, n))` materializes a **dense** `m × n` `Matrix{Float64}` and then stores
every entry as a structural nonzero. For an LV mesh at 3·10⁵ dofs and one chamber that is
a 2.4 MB dense intermediate and a fully populated off-diagonal block whose true support
is the endocardial surface dofs alone (≲ 1 % of rows). The cost recurs: `fill!(Jpd, 0.0)`
and `fill!(Jdp, 0.0)` (`rsafdq-operator.jl:60-61`) rewrite all of it every Newton
iteration, and `SchurComplementLinearSolver`'s `mul!(A₂₁z₁₊b₂, A₂₁, z₁)`
(`src/solver/linear/schur.jl:170`) runs over all of it too.

Note also `block_sizes[1] = ndofs(dh)` while `blocksizes(f)[1] = solution_size(f.structural_function) = ndofs(dh) + ndofs(lvh)`
(`src/modeling/functions.jl:28`). The two agree only because `ndofs(lvh) == 0` is enforced.

### 1.4 The two-pass assembly

`update_linearization!(op, residual_, u_, p)`, `rsafdq-operator.jl:36-85`:

- `:44-53` re-block `u_` and `residual_` as `BlockedVector`s from `blocksizes(J)`.
  `ud`, `up` and `Jdd` are bound and never used.
- `:60-61` hand-zero the two off-diagonal blocks (the assembler zeroes what it writes,
  which is `Block(1,1)` and the residual; `Block(2,2)` is never written and stays at the
  `fillzero!` value from setup).
- `:64-67` **pass 1**, volume:
  `start_assemble(strategy, J, residual)` → `FerriteOperators.AssembleLinearizationJR(assembler, u, p)`
  → `FerriteOperators.execute_on_subdomains!(task, strategy, subdomain_caches)`.
- `:71-82` **pass 2**, tying: for each chamber, for each `(sdh, tying_cache)`,
  `_update_tying_subdomain_Jr` (`:20-35`) runs a `FacetIterator` over
  `tying_cache.facets`, allocates `dofs = [celldofs(facet); pdof]` per facet, and calls
  the coupler's `assemble_facet!` **directly** with the FO 0.3 positional signature.
  Then `residualp[chamber_index] -= V⁰ᴰ` (`:79`).
- `:84` `FerriteOperators.finalize_assembly!(assembler)`.

`chamber_pressure` (`:73`) is computed only to be printed at `:81`; it indexes the
*blocked* `u` with a flat global index, which happens to be right only because
`pressure_dof_index_local == ndofs(dh) + i` and the first block has exactly `ndofs(dh)`
entries.

### 1.5 The 0D rows

The chamber row is `r[p] = ∫_Γ V³ᴰ(u) dΓ − V⁰ᴰ`. The integral half is written by the
coupler kernel (`fsi.jl:171`); the `− V⁰ᴰ` half is a post-hoc scalar subtraction in the
operator (`rsafdq-operator.jl:79`). `K[p, p]` is deliberately left zero
(`fsi.jl:183`, `# Kₑ[pdof, pdof] += 0`) — that zero (2,2) block is the structural
assumption `SchurComplementLinearSolver` is built on (`schur.jl:7-24`).

### 1.6 Where `V⁰ᴰ` comes from

- **Initialized** at `rsafdq2022.jl:201-202` to
  `compute_chamber_volume(dh, zeros(ndofs(dh)), setname, tying)` — the volume of the
  *undeformed* configuration. It is a placeholder that survives only until the first
  forward sync.
- **Updated** at `src/ferrite-addons/transfer_operators.jl:188`, inside
  `OS.forward_sync_external!(::VolumeTransfer0D3D)`:
  `chamber.V⁰ᴰval = outer_integrator.u[chamber.V⁰ᴰidx_global]`. `V⁰ᴰidx_global` is
  `num_unknowns_structure + num_unknown_pressures(circuit_model) + chamber_volume_idx_lumped`
  (`rsafdq2022.jl:199`) — an index into the *outer* operator-splitting vector.
- `V⁰ᴰval` is the one non-`const` field of the otherwise immutable
  `RSAFDQ2022SingleChamberTying` (`rsafdq2022.jl:11`).

So within one 3D solve **`V⁰ᴰ` is constant solver-supplied data, not an unknown.** In FO
terms it is a parameter. The reverse direction is `PressureTransfer3D0D`
(`transfer_operators.jl:206-216`): `inner_integrator.p[chamber.pressure_parameter_index_local] = outer_integrator.u[chamber.pressure_dof_index_global]`.

### 1.7 What the unimplemented stubs imply

Two methods `error("Not implemented yet.")`:

- `update_linearization!(op, u_, p)` — matrix-only (`:13-19`).
- `residual!(op, residual_, u_, p)` — residual-only (`:86-93`).

These are **not** dead code. `NewtonRaphsonSolver` has `simplified_newton::Bool = false`
(`src/solver/nonlinear/newton_raphson.jl:67`), and at `:234` the `simplified` branch calls
`residual!(op, residual, u, p)`. `MultiLevelNewtonRaphsonSolver` does the same at
`src/solver/nonlinear/multilevel_newton_raphson.jl:308`. So a 3D-0D model run with
`simplified_newton = true` errors out at the second Newton iteration. The matrix-only
entry point is used by the Newmark path only, which RSAFDQ never takes.

Reading: the operator supports exactly one entry point — the fused
`update_linearization!(op, r, u, p)` — and the coupled model is therefore restricted to
full-Jacobian Newton. Nothing in the code says this was intended rather than unfinished;
the `error` message says "yet".

### 1.8 Constraint elimination

`eliminate_constraints_from_linearization!(cache, op, f::RSAFDQ20223DFunction)`,
`rsafdq2022.jl:262-279`, hand-rolls the block elimination:

```julia
# apply_zero!(getJ(op, Block(1,1)), residual_block, ch)   # FIXME crashes
apply!(getJ(op, Block(1, 1)), ch)
apply_zero!(residual_block, ch)
getJ(op, Block((1, 2)))[ch.prescribed_dofs, :] .= 0.0
getJ(op, Block((2, 1)))[:, ch.prescribed_dofs] .= 0.0
```

It shadows the generic blocked eliminator at `src/solver/nonlinear/nlsolve_common.jl:55-84`,
which does the same thing by a different route. Why the commented-out three-argument
`apply_zero!` crashed is **not determinable from the code** — the block extraction
`@view op.J[Block(1,1)]` returns the stored `SparseMatrixCSC` for a `BlockArray`, which
`apply_zero!` should accept. Do not assume the crash reproduces; re-check it before
relying on the finding.

### 1.9 Test coverage

`test/integration/test_fsi.jl` is the only test that exercises any of this. Entry testset
`:65`, driver `test_solve_contractile_ideal_lv_3D0D` at `:8`, invoked twice — hand-written
circuit at `:109` and MTK circuit at `:143`. Both are **single-chamber (LV)**. The
assertions are `retcode == ReturnCode.Success` (`:59`) and `integrator.u ≉ u₀` (`:60`).

There is **no unit-level coverage at all**: no test names `AssembledRSAFDQ2022Operator`,
`RSAFDQ20223DFunction`, `RSAFDQ2022TyingInfo`, `Pressure3D0DVolumeCouplerIntegrator` or
`V⁰ᴰval`; `test/test_transfer.jl` does not cover `VolumeTransfer0D3D` /
`PressureTransfer3D0D`. The tutorial
`docs/src/literate-tutorials/cm03_3d0d-coupling.jl` is the second end-to-end consumer.

---

## 2. Mapping table

### 2.1 The structural prerequisite: chamber pressures move into the `DofHandler`

Everything else keys on this. Today the pressure dofs are index arithmetic outside
Ferrite; after the port they are `AlgebraicVariable`s inside `dh`.

```julia
# in semidiscretize, before close!(dh)
for coupling in coupler.chamber_couplings
    add!(dh, coupling.pressure_symbol_3D, AlgebraicVariable())
end
close!(dh)
```

**(a) Dof numbering.** Ferrite numbers algebraic dofs after every spatial dof
(`Ferrite docs/src/topics/algebraic_variables.md`, "Threading and matrix structure"), so
`ndofs(dh)` grows by the chamber count and the two-block split `[spatial | algebraic]`
holds **without renumbering** — which is precisely the layout the existing
`SchurComplementLinearSolver` wants. `pressure_dof_index_local` and
`pressure_dof_index_global` are both replaced by
`only(algebraic_dofs(dh, chamber.pressure_symbol))`, queried *after* the last
`renumber!`. Numerically the resulting global dof numbers are identical to today's
`ndofs_old(dh) + i`, which is what makes the equivalence gates in §4 entry-for-entry
comparisons rather than permuted ones.

**(b) Solution-vector layout.** `solution_size(::QuasiStaticFunction) = ndofs(dh) + ndofs(lvh)`
(`functions.jl:28`) now already contains the pressures, so the 3D block's vector becomes
`[u | p | q]` and its matrix `[u | p]` — the two agree on their common prefix and the
internal-variable tail is a clean suffix, exactly FO's own `[ū | q_cells | q_items]`
convention. **Reason 1 of `_check_rsafdq_internal_variables` dies.** Reason 2 (continuation
never advances internal variables) survives untouched, and a *new* reason appears — see
[§5.3](#53-condensed-elements-on-a-subdomain-declaring-global_dofs-are-rejected). Keep the
guard; rewrite its docstring.

`RSAFDQ20223DFunction.blocksizes` (`rsafdq2022.jl:99`) and `blocks` (`:107`) become a view
split of one vector rather than a concatenation of two. `solution_variables`
(`:112-118`) substitutes `algebraic_dofs` for `pressure_dof_index_local`.
`V⁰ᴰidx_global` (`:199`) shifts by whatever `solution_size(structural_fun)` changes by —
it is an offset into the *outer* split vector and must be recomputed, not carried.

**(c) Constraint handling.** `ch` is built on `dh`, so its sizing follows automatically.
Ferrite refuses Dirichlet conditions on algebraic names by design ("Scope"), which is the
right behaviour: no existing BC names a pressure. `apply_zero!(getJ(op), residual, ch)`
now runs over the *whole* matrix including the algebraic rows and columns, which is what
retires the hand-rolled block zeroing of §1.8 — it needs Ferrite's `BlockMatrix` support
in `ext/FerriteBlockArrays.jl:132-186` (`zero_out_columns!`, `zero_out_rows!`,
`add_inhomogeneities!`), which the `do/block-constraint-application` branch provides and
registered Ferrite 1.6 does not.
A *prescribed* chamber pressure (the `lv_pressure_given = true` circuit flavour) becomes
`add!(ch, AffineConstraint(only(algebraic_dofs(dh, :pₗᵥ)), Pair{Int,Float64}[], value))`
rather than a separate model path. Optional; not required by the port.

**(d) Sparsity.** The coupling entries are the caller's declaration
(`FerriteOperators/docs/src/elements.md`, "The sparsity is the caller's declaration"):

```julia
coupling = FacetCoupling(chamber.facets; algebraic_coupling = ((:u, chamber.pressure_symbol),))
spec     = StandardOperatorSpecification(; algebraic_couplings = (coupling,), constraint_handler = ch)
```

`FacetCoupling` — not `CellCoupling` — is the modelling-true statement here and is what
makes the off-diagonal block sparse instead of the dense placeholder of §1.3. (FO's own
tying fixture uses `CellCoupling` at `test/fixture_elements.jl:591` because its testbed is
a 3×3 toy where the distinction does not matter.) No `AlgebraicCoupling` is needed: the
`(p, p)` diagonal is always allocated, and RSAFDQ's chamber rows do not couple chambers to
each other on the 3D side.

### 2.2 Mechanism → seam → change

| hand-rolled mechanism | FO / Ferrite seam | Thunderbolt-side change |
|---|---|---|
| pressure index arithmetic, `rsafdq2022.jl:184`; twin fields `:4-5` | Ferrite `AlgebraicVariable` + `algebraic_dofs` | §2.1; delete both index fields from `RSAFDQ2022SingleChamberTying` |
| `dofs = [celldofs(facet); pdof]`, `rsafdq-operator.jl:27`; `pressure_dof = ndofs_per_cell(sdh)+1`, `fsi.jl:102` | `global_dofs(m, sdh)` + `global_dof_range(m, sdh)` (`FO src/core/element_interface.jl:80,89`) | `FerriteOperators.global_dofs(m::Pressure3D0DVolumeCouplerIntegrator, sdh) = algebraic_dofs(sdh.dh, m.pressure_symbol)`; cache stores `range_p = global_dof_range(m, sdh)` |
| `_update_tying_subdomain_Jr` facet loop, `:20-35` | `facet_items(m, sdh)` + `setup_facet_item_cache(m, sdh)` (`FO src/core/element_interface.jl:218,232`) | declare `m.facets`; FO owns the traversal, the per-item local system and the scatter |
| `_find_sdhs(dh, facetset)`, `:98-107` (guesses the owning subdofhandler from the *first* facet only) | FO resolves owning cells per subdomain; a facet whose cell is not in `sdh.cellset` is a **setup error** (`FO test/test_facet_items.jl:195-198`) | delete |
| `filter(facet -> facet[1] ∈ sdh.cellset, all_facets)`, `fsi.jl:107` | the declared set is the gate; per-subdomain restriction is FO's | delete the filter |
| `setup_3D0D_coupling_integrator`, `:109-144` (two methods, one with `# FIXME how to get around this fallacy here?`) | ordinary `setup_facet_item_cache` dispatch | delete both |
| `is_facet_in_cache`, `fsi.jl:112-116` | not consulted on the facet-item route (`elements.md`, "The declared set IS the gate") | delete — dead already, since the hand loop iterates the set directly |
| `assemble_facet!(Kₑ, rₑ, uₑ, cc, lfi, cache, t)`, `fsi.jl:118` (FO **0.3** signature) | `assemble_facet!(req, cache, args::FacetArgs, lfi::Int)` | rewrite the head; the body changes only `Kₑ→req.K`, `residualₑ→req.r`, `uₑ→args.states.u`, `pdof→first(c.range_p)`. Pattern: `FO test/fixture_elements.jl:550-577` |
| `residualp[chamber_index] -= V⁰ᴰ`, `:79` | `algebraic_items` + `setup_algebraic_cache` + `assemble_algebraic!` (`FO src/core/element_interface.jl:273,286`) | one item per chamber, dofs `[only(algebraic_dofs(dh, sym))]`; `req.r[1] -= V⁰ᴰ` |
| `chamber.V⁰ᴰval` read inside assembly, `:72` | `query_cell_parameters(cache, item::AlgebraicItem, p)` (algebraic items go through the cell query — `FO src/core/algebraic-task.jl:151`) | move V⁰ᴰ into the parameter bag; see [§5.4](#54-v⁰ᴰ-must-not-ride-a-cache-field) |
| `Jblock[Block(i,j)] = sparse(ones(...))`, `:175-177` | `algebraic_couplings` descriptor on the operator specification | one `FacetCoupling` per chamber |
| `BlockArray(spzeros(...))` + `Ferrite.fillzero!`, `:172-178` | `BlockedOperatorSpecification([n_u, n_chambers], BlockMatrix{Float64, Matrix{SparseMatrixCSC{Float64,Int}}}; algebraic_couplings, constraint_handler)` (`FO src/core/strategy.jl:40`) | replaces the whole hand-built block matrix; **name CSC blocks, not the CSR of the docs example** — `UMFPACKFactorization` in `schur.jl:84` needs CSC |
| two-pass `update_linearization!`, `:36-85` | one sweep; `update_linearization!(op, r, u, p)` on the ordinary operator | delete the method |
| `residual!` stub, `:86-93` | `evaluate!(op, r, u, p)` — inherited, no method needed | delete the stub; `simplified_newton = true` starts working |
| `update_linearization!(op, u, p)` stub, `:13-19` | inherited | delete the stub |
| hand block elimination, `rsafdq2022.jl:262-279` | `apply_zero!(K::BlockMatrix, f, ch)` via `Ferrite ext/FerriteBlockArrays.jl` | delete the specialization; the generic `nlsolve_common.jl:16` takes over |
| `getJ(op, ::Block)`, `:96` | `op.J` *is* a `BlockMatrix`; `getJ(op) = op.J` from `solver/interface.jl:9` suffices | delete the two-argument method — but check `nlsolve_common.jl:69-79` first, it is the only other consumer of a two-argument `getJ` and it becomes unreachable for RSAFDQ once the specialization above is deleted |
| `FerriteOperators.setup_operator_strategy_cache` `:156`, `create_system_matrix` `:158`, `setup_subdomain_caches` `:159`, `AssembleLinearizationJR` `:65`, `execute_on_subdomains!` `:67`, `finalize_assembly!` `:84` | `setup_operator(strategy, integrator, dh)` | delete all six. `AssembleLinearizationJR` **no longer exists** in FO 0.4; `setup_subdomain_caches` changed arity (3 positional → 5 positional + 3 keyword, `FO src/operators/setup.jl:196`) |

---

## 3. What dies

Deletions in `src/discretization/rsafdq-operator.jl` — **the whole file, all 189 lines**:

- `AssembledRSAFDQ2022Operator` (`:2`) and with it the last subtype of
  `FerriteOperators.AbstractBlockOperator`.
- both `update_linearization!` methods (`:13`, `:36`) and the `residual!` stub (`:86`).
- `_update_tying_subdomain_Jr` (`:20`).
- `getJ(op)` / `getJ(op, ::Block)` (`:95-96`).
- `_find_sdhs` (`:98`).
- `setup_3D0D_coupling_integrator`, both methods (`:109`, `:122`).
- `setup_stage_operator(::RSAFDQ20223DFunction, ::HomotopyPathSolver, …)` (`:146`) — the
  generic method at `homotopy.jl:98` takes over, given `get_strategy` and a `dh` accessor
  for `RSAFDQ20223DFunction`.
- the FO-internal reach-throughs listed in the last row of §2.2 (six symbols, all in
  this file).
- the `# TODO we are missing a way to dynamically extend the sparsity pattern in
  FerriteOperators` (`:157`) and `# TODO this is also not possible yet` (`:160`) — both
  are answered by `algebraic_couplings`.

Elsewhere:

- `src/modeling/coupler/fsi.jl:112-116` — `is_facet_in_cache`.
- `src/modeling/coupler/fsi.jl:107` — the `sdh.cellset` filter in `setup_boundary_cache`;
  the whole method becomes `setup_facet_item_cache`.
- `src/modeling/rsafdq2022.jl:262-279` — `eliminate_constraints_from_linearization!` for
  `RSAFDQ20223DFunction`.
- `src/modeling/rsafdq2022.jl:4-5` — `pressure_dof_index_local` / `_global`.
- `src/Thunderbolt.jl:59` — the `AbstractBlockOperator` import.

Retained deliberately:

- `compute_chamber_volume` (`rsafdq2022.jl:22-63`) — see [§5.1](#51-facet-functionals-do-not-exist).
- `_check_rsafdq_internal_variables` (`rsafdq2022.jl:163`) — reason 1 dies, reasons 2 and
  the new §5.3 reason stand. Rewrite the docstring, do not delete the guard.
- `src/solver/linear/schur.jl:199-218` `inner_solve_schur(J::BlockMatrix, r)` — already
  dead (no call site anywhere in the repo). Not this port's business, but worth removing
  while the file is open.

---

## 4. Step sequence

Each step is a separate change set, each ends green, each has a numeric gate.

### The ordering conflict, stated up front

The natural reading of "monolithic CSC first, blocked spec last as a solver optimization"
**does not work as written**. `SchurComplementLinearSolver` dispatches on
`A::AbstractBlockMatrix` (`schur.jl:58`), factorizes `@view A[Block(1,1)]` with UMFPACK
(`:84`), and reads `A[Block(1,2)]`, `[Block(2,1)]`, `[Block(2,2)]` (`:117-122`). A
monolithic `SparseMatrixCSC` has none of that, and wrapping it in a `BlockedArray` gives
`SubArray`s that UMFPACK will not factorize. The blocked specification is therefore **not
an optimization** — it is what keeps the shipped solver working.

The fix is cheap and keeps the ordering: for steps 1-3 the FSI test switches its inner
solver from `SchurComplementLinearSolver(UMFPACKFactorization())` to a plain
`UMFPACKFactorization()` over the full monolithic saddle-point system (indefinite, but
UMFPACK handles it), and step 4 restores it. That buys the real benefit of the
monolithic-first ordering: "does the assembly produce the same numbers" is separated from
"does the block plumbing work".

### Step 0 — prerequisite: the package-wide FO 0.4 migration

Not part of this port; it gates it. Blast radius, by qualified `FerriteOperators.` hit
count: `src/modeling/solid/elements.jl` (34), `test/test_elements.jl` (24),
`src/modeling/core/multi-integrator.jl` (13), `src/discretization/rsafdq-operator.jl` (10).
Symbols Thunderbolt uses that **do not exist in FO 0.4 at all**: `assemble_element!`,
`residual!`, `assemble_element_gto1!`, `query_element_parameters`,
`load_element_unknowns!`, `store_condensed_element_unknowns!`, `strategy_needs_atomic`,
`AssembleLinearizationJR`, `GenericFirstOrderTimeParameters`,
`GenericFirstOrderTimeElementParameters`, `AbstractGenericFirstOrderTimeVolumetricElementCache`,
`AbstractGenericFirstOrderTimeSurfaceElementCache`. Also `CudaDevice` regresses from
exported to internal (`ext/CuThunderboltExt.jl:29`). Follow
`FerriteOperators/docs/src/migration.md`; its ⚠ marks are the silent ones.

**Recommendation: repair the RSAFDQ operator minimally in step 0 rather than porting it
there.** Three edits — `AssembleLinearizationJR` → the 0.4 task construction,
`setup_subdomain_caches` 3-arg → 5-arg, `residual!` → `evaluate!` — plus the
`Pressure3D0DVolumeCouplerIntegrator` kernel signature. It is more work than deleting the
file, but every gate in steps 1-4 compares against the old operator, and the old operator
has to run under FO 0.4 for that to be possible. The alternative — port and repair in one
change set — leaves the FSI test with no reference to compare against, which is exactly
the situation this plan exists to avoid.

**Gate:** whole suite green under dev'ed FO and dev'ed Ferrite, RSAFDQ path unchanged in
behaviour.

⚠ `assemble_facet!` is the silent one. `Pressure3D0DVolumeCouplerCache` currently survives
on FO 0.3's signature only because `_update_tying_subdomain_Jr:32` calls it *by hand*.
Fused-route boundary caches are **not** validated at setup — `validate_element_cache` and
`validate_facet_item_cache` exist (`FO src/operators/setup.jl:285`,
`src/core/facet-task.jl:355`), `setup_boundaries` has no counterpart — so any other cache
in Thunderbolt whose `assemble_facet!` keeps the old signature will silently contribute
nothing. `src/modeling/core/weak_boundary_conditions.jl` alone has 23 `assemble_facet!`
uses. Grep every definition.

### Step 1 — algebraic-variable dofs, monolithic spec, hand loop retained

- `add!(dh, pressure_symbol, AlgebraicVariable())` per chamber before `close!` (§2.1).
- Replace `pressure_dof_index_local/_global` with `algebraic_dofs`; recompute
  `V⁰ᴰidx_global`; fix `solution_variables`, `blocksizes`, `blocks`, and both transfer
  operators.
- `strategy = AssemblyStrategy(FullAssembly(StandardOperatorSpecification(; algebraic_couplings, constraint_handler = ch)), SequentialScheduling(), device)`.
- The operator becomes an ordinary `setup_operator` result for the volume, plus a
  transitional wrapper that runs the tying loop as a **second assembler pass** on the same
  matrix:

  ```julia
  update_linearization!(w.op, r, u, p)                    # FO owns pass 1; zeroes J and r
  a = start_assemble(w.op.J, r; fillzero = false)         # Ferrite, not an FO internal
  # ... the existing facet loop, dofs = [celldofs(facet); algebraic_dofs(dh, sym)] ...
  # ... r[pdof] -= V⁰ᴰ ...
  ```

  This is Ferrite's own documented manual pattern
  (`Ferrite docs/src/topics/algebraic_variables.md`, "Assembly"), and it drops **all six**
  FO-internal reach-throughs at this step.
- Test-side: FSI test's inner solver → `UMFPACKFactorization()`.

**Gate:** with the equivalence harness of §7 on a pinned `(u, p, V⁰ᴰ)`,
`Matrix(J_new) ≈ Matrix(J_old)` and `r_new ≈ r_old`, entry for entry — the global dof
numbering is unchanged by construction (§2.1a). Plus FSI test green.

### Step 2 — facet items for the tying term

- Port the kernel to `assemble_facet!(req, cache, args::FacetArgs, lfi::Int)` for
  `ResidualRequest`, `JacobianRequest{:u}` and `JacobianResidualRequest`, with
  `provides_analytic(::Type{<:Pressure3D0DVolumeCouplerCache}, ::Union{JacobianKind{:u}, JacobianResidualKind}) = true`.
  Model: `FO test/fixture_elements.jl:521-577`.
- Declare `global_dofs`, `facet_items`, `setup_facet_item_cache`; forward all three from
  `NonlinearMultiDomainIntegrator2` (`src/modeling/core/multi-integrator.jl`), mirroring
  FO's own router at `FO src/elements/domain_elements.jl:68-78`.
- Delete the transitional wrapper's facet loop, `_find_sdhs`,
  `setup_3D0D_coupling_integrator`, `is_facet_in_cache`.

**Gate:** `Matrix(op.J) ≈` and `r ≈` step 1's values. Additionally
`check_derivatives(op, states, p, ctx)` (`FO docs/src/operators.md`, "Verifying derivative
implementations") on the ported kernel — the analytic tangent at `fsi.jl:153-180` has
never had an independent referee.

### Step 3 — the 0D rows as algebraic items

- `algebraic_items(m, dh) = [[only(algebraic_dofs(dh, c.pressure_symbol))] for c in chambers]`
  and a `setup_algebraic_cache` returning one cache serving all chambers, keyed by
  `args.item.index`.
- Residual kernel: `req.r[1] -= V⁰ᴰ(args.p, args.item.index)`. Nothing to declare for the
  Jacobian — the row is constant in `u`, so the AD fallback produces the correct zero
  block, and the diagonal entry is always allocated.
- Move V⁰ᴰ into the parameter bag (§5.4). Delete the residual wrapper entirely; the
  operator is now `setup_operator(...)` unmodified.

**Gate:** `Matrix(op.J) ≈` and `r ≈` step 2's values.

### Step 4 — blocked specification, Schur restored

- `BlockedOperatorSpecification([n_u, n_chambers], BlockMatrix{Float64, Matrix{SparseMatrixCSC{Float64, Int}}}; algebraic_couplings, constraint_handler = ch)`
  where `n_u = ndofs(dh) - n_chambers`.
- Restore `SchurComplementLinearSolver(UMFPACKFactorization())` in the test and tutorial.
- Delete `eliminate_constraints_from_linearization!(…, ::RSAFDQ20223DFunction)` and the
  two-argument `getJ`.

**Gate:** `Matrix(op.J) ≈` step 3's monolithic J; `op.J isa BlockMatrix`;
`blocklengths(axes(op.J, 1)) == [n_u, n_chambers]`; FSI test green with Schur.
Pattern: `FO test/test_blocked_assembly.jl:50-76`.

### Step 5 — sparsity and cleanup

- `CellCoupling` → `FacetCoupling(chamber.facets; …)` if step 1 took the conservative
  route. **Gate:** `Matrix(op.J) ≈` step 4's, and `nnz(op.J[Block(1,2)])` drops by orders
  of magnitude.
- Delete `AbstractBlockOperator` from the import list; file the FO-side removal.
- Rewrite `_check_rsafdq_internal_variables`'s docstring (§2.1b, §5.3).

---

## 5. Known gaps and fallbacks

### 5.1 Facet functionals do not exist

Chamber volume is a surface integral, `V³ᴰ = ∫_Γ …`. FO's reduction family
(`evaluate_functional`) reaches cell items via `evaluate_cell_functional` and algebraic
items via `evaluate_algebraic_functional`, and **facet items contribute nothing** — stated
in the three-families table of `FO docs/src/elements.md` and asserted structurally in
`FO test/test_facet_items.jl:321-329` (`_may_contribute(fd, FunctionalKind(:probe)) == false`).

**Fallback: keep `compute_chamber_volume` (`rsafdq2022.jl:22-63`) solver-side.** This costs
nothing today: it is called exactly once, at `rsafdq2022.jl:202`, for the reference volume.
It becomes binding the moment someone wants V³ᴰ as a per-step output, or wants the
reference volume computed at a non-zero configuration.

**This is the expected consumer-driven ask on FO**: an `evaluate_facet_functional` hook
next to `evaluate_cell_functional`, keyed on `(FunctionalKind, cache, args, lfi)`.

While that function stays: its subdomain lookup at `rsafdq2022.jl:30` is
`sdhi = typeof(cell) == Hexahedron ? 1 : min(2, length(dh.subdofhandlers))`, carrying a
`# :)` and a comment admitting the right way. It is a guess that silently picks the wrong
interpolation on any mesh whose subdofhandler order differs from the assumption. Fix it
independently of the port, or let a facet functional retire it.

### 5.2 Multi-domain routing of `algebraic_items` — **not a blocker here**

`algebraic_items(integrator, dh)` is one declaration per integrator over the whole
`DofHandler` (`FO src/core/element_interface.jl:273`, resolved once at
`src/core/algebraic-task.jl:379`). FO's own `AnyMultiDomainIntegrator` forwards
`global_dofs`, `facet_items`, `setup_facet_item_cache`, `setup_element_cache` and
`setup_boundary_cache` (`FO src/elements/domain_elements.jl:68-85`) but **not**
`algebraic_items` / `setup_algebraic_cache`. A routed operator therefore falls to the
default `()` and its 0D rows **silently vanish**.

Thunderbolt is not exposed: it uses its own `NonlinearMultiDomainIntegrator2`
(`src/modeling/core/multi-integrator.jl:1`) and can define the two methods itself.
And Thunderbolt's chamber set does not need routing — one algebraic cache serves every
chamber, `args.item.index` selects, and all items are uniformly sized (one dof each,
which is FO's uniform-size requirement). Routing would only be needed if chambers had
*different* 0D row physics; RSAFDQ's do not.

Worth filing against FO anyway, because the failure mode is silence rather than an error.

### 5.3 Condensed elements on a subdomain declaring `global_dofs` are rejected

`FO src/elements/ad_element.jl:254-264`: a cache with `has_internal_state` on a subdomain
declaring `global_dofs > 0`, without an analytic `Consistent` Jacobian kernel, throws at
`setup_operator`. The reason is real — the generic corrector block spans the field space
while the AD partials span the augmented system.

`global_dofs` is declared **per integrator per subdomain and shared by that subdomain's
volumetric and boundary kernels** (`FO docs/src/elements.md`, "Elements with global
dofs"). There is no way to declare the pressure dof "for the facet item only". So
declaring it for the tying term on the myocardium subdomain makes that subdomain's
*volumetric* element carry the augmented tail — and a condensed myocardium material
without an analytic tangent is then rejected at setup.

Today `_check_rsafdq_internal_variables` bans such materials anyway, so nothing breaks.
But it means **the port trades reason 1 of that ban for a new one**, and lifting the ban
later needs one of:

- the myocardium element serving `JacobianResidualKind` / `JacobianKind{:u}` analytically
  (Thunderbolt has analytic tangents for some materials — verify per material, this was
  not checked). Note this layer is under active change in FO: the admissibility rule is
  moving from `provides_analytic` to a `serves_kind` predicate that also counts the
  decorator's generic completions, so re-read `ad_element.jl` before writing against it;
  or
- **the second predicted FO ask**: a per-item-family `global_dofs` declaration, so a facet
  item can carry an augmented tail that the cell family of the same subdomain does not.

### 5.4 `V⁰ᴰ` must not ride a cache field

Today `chamber.V⁰ᴰval` is mutated from outside the operator
(`transfer_operators.jl:188`) and read inside assembly (`rsafdq-operator.jl:72`). Carrying
that mutable chamber struct into an algebraic cache reproduces exactly the
solver-state-smuggling pattern `FO docs/src/migration.md` ("Per-worker mutable state")
calls out, and interacts badly with `duplicate_for_device`: whether a worker's duplicate
sees a later write depends on whether the duplication copies the reference or the value —
**not verified**, and not something to depend on either way.

Route it through the parameter bag instead: algebraic items get `args.p` from
`query_cell_parameters(cache, item::AlgebraicItem, p)` (`FO src/core/algebraic-task.jl:151`),
so `query_cell_parameters(c::ChamberCache, item, p) = p.V⁰ᴰ[item.index]` is the whole
change. Complication: `HomotopyPathSolver` currently passes the bare pseudo-time as `p`
(`homotopy.jl:126-129`), so the continuation's parameter object has to grow a field —
which is a small change to `FullStateStage`, not to the solver.

Second-order benefit: with V⁰ᴰ in `p`, `update_parameter_jacobian!` gives ∂r/∂V⁰ᴰ for free,
which is the sensitivity a monolithic 3D-0D Newton would want.

### 5.5 Assembly strategies narrow

`global_dofs` rejects `ColoredScheduling` (`FO src/operators/setup.jl:224`), the
`ElementAssembly` form (`:229`) and patch assembly. The FSI test uses
`SequentialAssemblyStrategy`, so nothing breaks there — but `PerColorAssemblyStrategy` and
`ElementAssemblyStrategy` become setup errors for any 3D-0D model. **Check what
`FiniteElementDiscretization` defaults to before step 1**; this was not verified. The
parallel route stays open: `SequentialScheduling()` under a `PolyesterDevice`, whose atomic
scatter resolves the shared dof (`FO test/test_facet_items.jl:310-317`).

### 5.6 The Ferrite dependency

`AlgebraicVariable`, `algebraic_dofs`, `AlgebraicValues`, `CellCoupling` /
`FacetCoupling` / `AlgebraicCoupling`, the `algebraic_couplings` keyword, and
`apply!(::BlockMatrix, …)` are all on `Ferrite#do/block-constraint-application` and in no
registered Ferrite. `Project.toml:70` pins `Ferrite = "1.6"`.

Capability gating in the meantime — FO's own pattern, worth copying verbatim:

```julia
if !isdefined(Ferrite, :AlgebraicVariable)
    @info "Skipping …: this Ferrite has no `AlgebraicVariable`"
else
    …
end
```

(`FO test/test_algebraic_items.jl:8-11`, `test/test_facet_items.jl:271-273`.) FO passes
`algebraic_couplings` to Ferrite only when non-empty (`FO src/operators/setup.jl:112-118`),
so an operator declaring none is unaffected on any Ferrite.

**What gates Thunderbolt CI adopting it:** a Ferrite release carrying the algebraic-variable
work (`Ferrite#master` `30a02c157`, "Algebraic variables (Global dofs 3) (#1482)", is
merged; the block/CSR constraint application on top is not) plus an FO 0.4 release with a
compat bound naming it. Until then Thunderbolt CI can only run the ported RSAFDQ path from
a manifest with two dev'ed paths — which is fine for local work and not fine for CI.
**Recommended sequencing:** do steps 0-5 on a branch with a dev'ed manifest, and hold the
merge until Ferrite ships. Do *not* try to keep both the old and the new operator alive
behind a capability gate: the dof layout differs, so the gate would have to reach into
`semidiscretize`, and every downstream index (§2.1e) would need two branches.

### 5.7 Smaller findings, port-independent

- `residual!` and matrix-only `update_linearization!` are `error("Not implemented yet.")`,
  reachable through `NewtonRaphsonSolver(; simplified_newton = true)` (§1.7). The port
  fixes both by inheritance.
- `rsafdq-operator.jl:48-49, 57` bind `ud`, `up`, `Jdd` and never use them.
- `rsafdq-operator.jl:73` computes `chamber_pressure` only for a `@debug` at `:81`, by
  indexing a `BlockedVector` with a flat global index.
- `rsafdq-operator.jl:22-24` allocates `Kₑ`, `rₑ`, `uₑ` per subdomain call and
  `dofs = [celldofs(facet); pdof]` per facet (`:27`), inside the Newton loop —
  the `# FIXME allocator api` and `# FIXME loader function` comments.
- `src/solver/linear/schur.jl:199-218` `inner_solve_schur` has no call site anywhere.
- `bak/examples/lv-with-mtk-circuit.jl:287` calls `ChamberVolumeCoupling` with three
  positional arguments against today's six-field struct — stale, but it is under `bak/`.
- The `# FIXME crashes` at `rsafdq2022.jl:272` (§1.8) has no diagnosable cause in the code.

---

## 6. Definition of done

1. `AssembledRSAFDQ2022Operator` is gone and nothing subtypes
   `FerriteOperators.AbstractBlockOperator` — the supertype's own "Slated for removal"
   warning (`FO src/operators/general.jl:166-175`) names this migration as its last
   remaining consumer, so FO can then delete it.
2. `src/discretization/rsafdq-operator.jl` is deleted; `setup_stage_operator` for
   `RSAFDQ20223DFunction` resolves to the generic continuation method at
   `homotopy.jl:98`.
3. **Zero `FerriteOperators.`-internal reach-throughs in the RSAFDQ path.** Concretely,
   `grep -n "FerriteOperators\." src/discretization/ src/modeling/coupler/fsi.jl` returns
   only public API. The six to remove are `setup_operator_strategy_cache`,
   `create_system_matrix`, `setup_subdomain_caches`, `AssembleLinearizationJR`,
   `execute_on_subdomains!`, `finalize_assembly!` (all in `rsafdq-operator.jl`), plus
   `EmptySurfaceElementCache` at `:143`.
4. The chamber pressures are `AlgebraicVariable`s in the `DofHandler`; no code computes a
   dof index by adding to `solution_size`.
5. `op.J isa BlockMatrix{Float64, Matrix{SparseMatrixCSC{Float64, Int}}}` with block
   lengths `[ndofs(dh) - n_chambers, n_chambers]`, and the off-diagonal blocks' sparsity
   is the declared `FacetCoupling`, not a dense placeholder.
6. `Project.toml` names an FO 0.4 bound and a Ferrite version carrying
   `AlgebraicVariable` and `apply!(::BlockMatrix, …)`; the FSI test is *not* capability
   gated (§5.6 — gate the branch, not the test).
7. `check_derivatives` passes on the ported coupler element.
8. `simplified_newton = true` works on a 3D-0D model.
9. `_check_rsafdq_internal_variables`'s docstring reflects the surviving reasons
   (§2.1b, §5.3), not the retired layout one.

---

## 7. Test strategy

### 7.1 The equivalence harness

The gates in §4 all have the same shape and want one helper, built in step 0 while the old
operator still runs:

```julia
# test/test_rsafdq_operator.jl  (new)
function rsafdq_reference_state(; seed = 42)
    # small ideal-LV mesh, one chamber, the test_fsi.jl model
    # returns (; f, op, u, p, V⁰ᴰ) with u a fixed pseudo-random vector, NOT a solved state
end

function assemble_pair(op, u, p)
    r = zeros(residual_size(op)); update_linearization!(op, r, u, p); (Matrix(getJ(op)), r)
end
```

and a pinned reference produced once, at step 0, from the old operator:
`(J₀, r₀) = assemble_pair(op_old, u, p)` serialized into `test/data/`. Every later step
asserts `J ≈ J₀` and `r ≈ r₀`. Use `≈`, never `==` — FO's facet items group by owning cell
and visit facets in a different order than the hand loop, so the summation into a shared
dof differs in the last bits (`FO test/test_facet_items.jl:56-58` makes the same point).

Pin at a **non-solution** state. A converged `u` has `r ≈ 0`, which makes the residual gate
vacuous; the Jacobian gate would still bite, but half the value of the harness is in the
coupling rows of `r`, and those are only nonzero away from equilibrium. Assert positively
that the coupling is exercised, following `FO test/test_facet_items.jl:297-299`:

```julia
@test maximum(abs, J₀[pdof, :]) > 0
@test maximum(abs, J₀[:, pdof]) > 0
@test abs(r₀[pdof]) > 0
```

Also pin `V⁰ᴰ` explicitly rather than letting it come from a transfer — otherwise the
harness depends on the 0D solve.

### 7.2 What gets pinned

| quantity | why |
|---|---|
| `Matrix(J)` at the fixed `(u, p, V⁰ᴰ)` | the whole point; catches every mapping error in §2.2 |
| `r` at the same state | catches the `− V⁰ᴰ` row and the `∫V³ᴰ` row separately from the Jacobian |
| `algebraic_dofs(dh, sym)` vs. the old `ndofs_old(dh) + i` | proves §2.1a — the numbering really is unchanged, which is what licenses entry-for-entry comparison |
| `nnz` per block, before and after step 5 | the sparsity tightening is otherwise invisible |
| the ported facet kernel against a hand-rolled Ferrite loop | independent of the operator; model `FO test/fixture_elements.jl:601-638` |

### 7.3 Existing coverage, and what it does not cover

`test/integration/test_fsi.jl` asserts only `retcode == Success` (`:59`) and
`integrator.u ≉ u₀` (`:60`). It will **not** catch: a wrong Jacobian that Newton still
converges through, a dropped coupling term that shifts the equilibrium slightly, a
silently-vanished 0D row while the pressure is otherwise pinned by the circuit, or the
dense-vs-sparse sparsity change. It is a smoke test, and after the port it will still be
the only end-to-end one. **The unit-level harness of §7.1 is the actual gate; the FSI test
is the backstop.**

Both FSI invocations are single-chamber. The port's multi-chamber behaviour — several
algebraic items sharing a partition, several `FacetCoupling` descriptors, block sizes
`[n_u, 2]` — is untested in either the old code or the new. Add a two-chamber fixture to
the harness even without a two-chamber circuit model; the operator can be assembled without
being solved.

Also worth adding while the file is open: `test/test_transfer.jl` covers neither
`VolumeTransfer0D3D` nor `PressureTransfer3D0D`, and both are rewritten by §2.1e.
