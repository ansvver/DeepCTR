
import  merge_train_data
from  merge_train_data import *
from Pytorch.model import DeepFM
import torch

field_size = 16
result_dict = merge_train_data.read_train_data('C:\conf_test\deepctr\data\deepfm-train-data', field_size)
# with torch.cuda.device(2):
#     deepfm = DeepFM.DeepFM(39,
#                            result_dict['feature_sizes'],
#                            verbose=True,
#                            use_cuda=True,
#                            weight_decay=0.0001,
#                            use_fm=True,
#                            use_ffm=False,
#                            use_deep=True).cuda()
#     deepfm.fit(result_dict['index'],
#                result_dict['value'],
#                result_dict['label'],
#                None,
#                None,
#                None,
#                ealry_stopping=True,
#                refit=True)

deepfm = DeepFM.DeepFM(field_size,
                        result_dict['feature_sizes'],
                        verbose=True,
                        use_cuda=False,
                        weight_decay=0.0001,
                        use_fm=True,
                        use_ffm=False,
                        use_deep=True)
deepfm.fit(result_dict['index'],
           result_dict['value'],
           result_dict['label'],
           None,
           None,
           None,
           ealry_stopping=True,
           refit=True)