# Tools for recording and analyzing for Ca data

installation: `pip install git+https://github.com/janclemenslab/ca_utils`

## Usage
```python
import ca_utils as ca

expt_id = '20190411_005'
root = '../../ca_img/dat/20190411/'
s = ca.Session(root + expt_id)
print(s.logs)

for trial in range(s.nb_trials):
    stack = s.stack(trial)
    print(stack.shape)
```
