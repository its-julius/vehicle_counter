import torch
print('Torch CUDA Current Device: {}'.format(torch.cuda.current_device()))
print('Torch CUDA Device: {}'.format(torch.cuda.device(torch.cuda.current_device())))
print('Torch CUDA Device Count: {}'.format(torch.cuda.device_count()))
print('Torch CUDA Device Name: {}'.format(torch.cuda.get_device_name(torch.cuda.current_device())))
print('Torch CUDA Availability: {}'.format(torch.cuda.is_available()))

