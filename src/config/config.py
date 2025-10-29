__all__ = ["Config"]


class Config:
    def __init__(self):
        self.data = {
            "x_load_path"       : "data/waveform/x.pt",
            "y_load_path"       : "data/waveform/y.pt",
            "split_load_path"   : "data/waveform/split.pt",
            "channel_perm"      : True,
            "channel_drop"      : True,
            "batch_size"        : 256,
            "num_workers"       : 8
        }
        self.model = {
            "D"                 : 256,
            # tokenizer
            "S"                 : 40,
            "stride"            : 20,
            # masking
            "p_point"           : 0.2, 
            "p_span_small"      : [0.0, 0.5],   # channel = 1
            "p_span_large"      : [0.0, 1.0],   # channel > 1
            "p_hide"            : 0.9, 
            "p_keep"            : 0.1,
            # embedding
            "C_max"             : 8,
            "L_max"             : 1024,
            # transformer
            "num_layers"        : 4,
            "nhead"             : 8,
            "dim_feedforward"   : 1024,
            # freeze
            "freeze_embedding"  : False,
            "freeze_transformer": 0
        }
        self.runner = {
            # loss
            "enable"            : [True, True, False],
            "weight"            : [ 0.2,  0.8,   0.0],
            "T"                 : 0.2,
            # optimizer
            "lr"                : 0.005,
            "step_size"         : 20,
            "gamma"             : 0.98
        }
        self.trainer = {
            "max_epochs"        : 10000,
            "log_save_fold"     : "log/",
            "ckpt_save_fold"    : "ckpt/",
            "monitor"           : "loss/valid",
            "save_top_k"        : 10,
            "ckpt_load_path"    : None,
            "resume"            : False
        }
