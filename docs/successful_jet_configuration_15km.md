# Successful 15 km Upper-Level Jet Configuration

Recorded: 2026-09-03
Status: User-designated most successful configuration
Associated output: `cm1out_25N_9o_jet_30_15km.nc`

## Authoritative namelist values

```fortran
&jet_params
  add_westerly_jet = .true.,
  jet_umax   = 30.0,
  jet_z_ctr  = 15000.0,
  jet_z_rad  = 2000.0,
  jet_y_dist = 999000.0,
  jet_y_rad  = 325000.0,
  jet_z_bot  = 5000.0,
  jet_z_top  = 25000.0,
/
```

## Physical interpretation

| Parameter | Value | Meaning |
|---|---:|---|
| `add_westerly_jet` | `.true.` | Enable the balanced westerly jet |
| `jet_umax` | 30 m s^-1 | Maximum jet wind speed |
| `jet_z_ctr` | 15 km | Jet centre height |
| `jet_z_rad` | 2 km | Vertical radius |
| `jet_y_dist` | 999 km | Meridional distance from the TC centre, approximately 9.0 degrees latitude |
| `jet_y_rad` | 325 km | Meridional radius, approximately 2.93 degrees latitude |
| `jet_z_bot` | 5 km | Lower cutoff height |
| `jet_z_top` | 25 km | Upper cutoff height |

## Consistency note

The screenshot comments do not match two numerical values:

- `jet_y_dist = 999000 m` corresponds to approximately 9 degrees, not `8 deg x 111 km`.
- `jet_y_rad = 325000 m` corresponds to approximately 2.93 degrees, not `4 deg x 111 km`.

For reproduction, the numerical namelist values above are authoritative.
