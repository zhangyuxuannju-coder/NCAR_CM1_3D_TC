import sys
from pathlib import Path
source=Path('scripts/complete_se_source_details.py').read_text()
source=source.replace("source=Path('scripts/run_complete_se_attribution.py').read_text()","source=Path('output/se_boundary_fix_validation_new01/complete_attribution_boundary_fixed.py').read_text()")
source=source.replace("REFERENCE=Path('output/se_complete_forcing_operator_window3h_new01')","REFERENCE=Path('output/se_complete_boundary_fixed_window3h_new01')")
exec(compile(source,'existing_source_details_corrected_baseline','exec'),globals())
