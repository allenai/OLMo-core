import torch
from olmo_core.utils import get_or_init_stream
    
@torch.compiler.disable(recursive=False)         # helper runs eagerly, 
def async_copy_to_cpu(gpu_buf):
    # *** async copy to CPU for future GroupedGEMM ***
    # start a new stream for the copy
    dtoh_stream = get_or_init_stream(id=3, priority=-5) # TODO: check any id that's not 0?
    
    # Make the copy_stream start **after** everything already queued
    #  on the current stream (default) that touches batch_size_per_expert.
    dtoh_stream.wait_stream(torch.cuda.current_stream())
    
    with torch.cuda.stream(dtoh_stream):
        cpu_buf = torch.empty_like(gpu_buf,
                                device="cpu", pin_memory=True) # compile does not work with pin_memory
        cpu_buf.copy_(gpu_buf, non_blocking=True)
    
    dtoh_event = dtoh_stream.record_event()
    
    # cpu_buf = gpu_buf.to(torch.device("cpu"), non_blocking=True)
    # Keep the source tensor alive until the copy_stream is done
    # gpu_buf.record_stream(dtoh_stream) # NOTE: does not work with compile
    return cpu_buf, dtoh_stream, dtoh_event
        