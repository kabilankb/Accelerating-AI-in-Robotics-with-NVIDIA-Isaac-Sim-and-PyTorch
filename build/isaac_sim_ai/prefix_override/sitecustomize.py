import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/kabilankb/isaacsim_ai/install/isaac_sim_ai'
