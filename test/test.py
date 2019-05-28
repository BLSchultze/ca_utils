import ca_utils.io as io

date_name = '20190411'
session_number = 5
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
pattern = 'PPAU20'
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

trial = 1
print('split nothing: ', s.stack(trial, split_channels=False, split_volumes=False).shape)
print('split nothing, force dims: ', s.stack(trial, split_channels=False, split_volumes=False, force_dims=True).shape)
print('split channels:', s.stack(trial, split_channels=True, split_volumes=False).shape)
print('split volumes: ', s.stack(trial, split_channels=False, split_volumes=True).shape)
print('split channels and volumes: ', s.stack(trial, split_channels=True, split_volumes=True).shape)

print(f're-loading with pixel-wise zpos:')
s = io.Session(session_path, with_pixel_zpos=True)
print(s.log.zpos_frames[0].shape)
