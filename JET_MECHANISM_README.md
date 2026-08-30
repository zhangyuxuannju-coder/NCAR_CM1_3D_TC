# JET mechanism diagnostics

## Scientific interpretation

The workflow separates ventilation/shear, boundary-layer inertial stability,
isentropic energetics, and upper-level eddy/SE response. Outflow-layer
inertial stability is used only as a balanced dynamical coefficient. It is not
labelled an energetic resistance.

`F_lambda_env` means the total jet-induced eddy response
`eddy(JET) - eddy(CTRL)`, not a pure imposed-jet flux.

## Server command

From `/data1/home/zhangyx/project/TC_dynamic`:

```bash
/data1/home/zhangyx/miniconda3/envs/cm1_tc/bin/python \
  scripts/run_jet_mechanism_diagnostics.py \
  --ctrl /data/zhangyx/DATA/cm1out_25N_nojet.nc \
  --jet /data/zhangyx/DATA/cm1out_25N_9o_jet_30.nc \
  --output output/jet_mechanism_25N_ctrl_vs_jet30_9deg \
  --start-hour 30 --end-hour 84 --step-hour 2 \
  --energy-times 40,55,70,80 \
  --se-time 70 --f 6.2e-5 \
  --regularization 1e-5,1e-4,1e-3
```

The ventilation index uses `tcpyPI` and the 200-800-km mean environmental
sounding by default. A fixed independently calculated PI can be supplied with
`--potential-intensity-ms`.

Run `--preflight-only` first if a new data source is introduced. Missing
budget fields are listed in `preflight.json`; they are never replaced silently
by zero in the mechanism report.

## Output contract

- `preflight.json`: time coverage, dimensions and variable capability.
- `timeseries.csv`: aligned diagnostics from all four pathways.
- `data/*.npz`: reduced r-z fields at each processed time.
- `energetics/*`: isentropic streamfunctions, cycles and closure metadata.
- `se/*`: C0/CI/CF/JF products for each regularization value.
- `figures/*`: PNG and PDF summary figures.
- `lead_lag.csv`, `strength_matching.npz`, `mechanism_report.md`: attribution products.

A full heat-engine interpretation is withheld unless the isentropic mass
closure ratio is at most 0.10 and the resolved first-law residual is at most
0.20. Regularized SE output in raw non-elliptic regions is labelled a balanced
projection.
