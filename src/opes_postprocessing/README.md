## OPES Postprocessing
This module is an extension for . Utils contains all scripts for the postprocessing of OPES simulations.

## Example usage
```python
from opes_postprocessing.utils import calc_fes

calc_fes.from_state(state_data,
                    state_info,
                    cvs=None,
                    process='last',
                    mintozero=True,
                    temp=310,
                    units='kT',
                    bins=100,
                    device='mt',
                    verbose=True):
```

The scripts are based on the scripts from the original OPES paper, but rewritten to also be runable in parallel or on GPUs.