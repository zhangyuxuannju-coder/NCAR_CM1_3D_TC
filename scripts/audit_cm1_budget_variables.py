from netCDF4 import Dataset
import sys

for path in sys.argv[1:]:
    print(f"\nFILE {path}")
    with Dataset(path) as ds:
        print("DIMS", {k: len(v) for k, v in ds.dimensions.items()})
        if "time" in ds.variables:
            time = ds.variables["time"]
            print("TIME", time[:3].tolist(), time[-3:].tolist(), getattr(time, "units", ""))
        print("CORE VARIABLES")
        for name in ("u", "v", "w", "rho", "prs", "th", "psfc"):
            if name in ds.variables:
                var = ds.variables[name]
                print(name, var.dimensions, getattr(var, "units", ""), getattr(var, "long_name", ""))
        print("MOMENTUM BUDGET VARIABLES")
        for name in sorted(ds.variables):
            if name.startswith(("ub_", "vb_")):
                var = ds.variables[name]
                print(name, var.dimensions, getattr(var, "units", ""), getattr(var, "long_name", ""))
