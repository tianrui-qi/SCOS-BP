Since we put all `.ipynb` under `/notebook/` subfolder, Python working directory
is not the project root by default, which may cause import errors.
To fix this, add the following code at the beginning of each notebook before any
other imports:

```python
import sys, os
os.chdir(os.path.abspath(os.path.join(os.getcwd(), '..')))
project_root = os.getcwd()
if project_root not in sys.path: sys.path.append(project_root)
```
