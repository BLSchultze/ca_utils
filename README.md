# Tools for recording and analyzing for Ca data

installation: `pip install git+https://github.com/janclemenslab/ca_utils`

## Usage
```python
import ca_utils.io as io

date_name = '20190410'
session_number = 1
root = '../../ca_img/dat'
session_path = f'{root}/{date_name}/{date_name}_{session_number:03d}'
s = io.Session(session_path)
print(s.log)

for trial in range(s.nb_trials):
    stack = s.stack(trial)
    print(f'stack shape: {stack.shape}')

# find all trials for which the `stimFileName` column for the `left_sound` channel contains `PPAU60`
column_name = 'stimFileName'
channel = 'left_sound'
pattern = 'PPAU60'
op = 'in'
trials = s.argfind(column_name, pattern, channel, op)
print(f'matching trials: {trials}')
print(s.find(column_name, pattern, channel, op))

print(f'loading matching trials:')
for trial in trials:
    stack = s.stack(trial)

# find all trials for which the `silencePre` column for the `left_sound` channel equals 3000
column_name = 'silencePre'
channel = 'left_sound'
pattern = 3000
op = '=='
trials = s.argfind(column_name, pattern, channel, op)
print(f'matching trials: {trials}')
print(s.find(column_name, pattern, channel, op))
```

### Loading data into SIMA
First, initialize the session:
```python
import ca_utils.io as io

date_name = '20190410'
session_number = 1
root = '../../../ca_img/dat'
session_path = f'{root}/{date_name}/{date_name}_{session_number:03d}'
s = io.Session(session_path)
s.log
```
Then, load the data for all trials
```python
stacks = [s.stack(trial, split_channels=True, split_volumes=True) for trial in range(s.nb_trials)]
```
and create the SIMA data set:
```python
import sima
sequences = [sima.Sequence.create('ndarray', stack) for stack in stacks]
dataset = sima.ImagingDataset(sequences, 'example_np.sima', channel_names=['gcamp', 'tdtomato'])
```
