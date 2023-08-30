import os
import socket

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def get_summarywriter(out_dir):

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(
        out_dir, current_time + "_" + socket.gethostname()
    )
    return SummaryWriter(log_dir=log_dir)

