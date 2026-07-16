# Mixed-order collision split in v9

The v9 line-source experiment uses two angular quadratures:

```text
N_u: high angular order for the uncollided equation
N_c: lower angular order for the collided equation
```

The default paper configuration is `N_u=32` and `N_c=8,16`.

At each time step:

1. The previous total angular field is represented on the high-order grid.
2. The scattering-free uncollided equation is swept on the high-order grid.
3. Scalar flux and current are computed from the high-order uncollided field.
4. The isotropic collided source is assembled from the uncollided and collided scalar fluxes.
5. The collided equation is iterated on the low-order grid.
6. Moment-preserving quantization is applied only to the low-order collided angular field.
7. The collided history is transferred back to the high-order grid for time-step relabeling.

The low-to-high transfer uses nearest-direction reconstruction followed by an affine correction that enforces

```text
phi_low = phi_transferred
Jx_low  = Jx_transferred
Jy_low  = Jy_transferred
```

up to floating-point roundoff.

The line source is selected first because it isolates the intended mechanism without material-interface and boundary-inflow complications. Once this experiment is established, the lattice and hohlraum are appropriate follow-up tests.
