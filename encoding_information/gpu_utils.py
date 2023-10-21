import os        

def limit_gpu_memory_growth():
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'