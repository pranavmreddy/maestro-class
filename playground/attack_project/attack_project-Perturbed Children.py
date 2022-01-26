from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import DataLoader
from Maestro.attacker_helper.attacker_request_helper import virtual_model
from transformers.data.data_collator import default_data_collator
from Maestro.evaluator.Evaluator import get_data

class ProjectAttack:


    def __init__(self, vm, image_size: List[int], l2_threshold=7.5):
        self.vm = vm
        self.image_size = image_size
        self.l2_threshold = l2_threshold

    
    def zero_gradients(x):
        if isinstance(x, torch.Tensor):
            if x.grad is not None:
                x.grad.detach_()
                x.grad.zero_()
        elif isinstance(x, collections.abc.Iterable):
            for elem in x:
                zero_gradients(elem)


    def attack(self, original_image:  np.ndarray, labels: List[int], target_label: int, epsilon = 0.214):
        """
        args:
            original_image: a numpy ndarray images, [1,3,32,32]
            labels: label of the image, a list of size 1
            target_label: target label we want the image to be classified, int
        return:
            the perturbed image
            label of that perturbed iamge
            success: whether the attack succeds
        """

        perturbed_image = deepcopy(original_image)
        # --------------TODO--------------
        
        # write your attack function here

        print('target', target_label)

        target_labels = [target_label]*len(labels)

        print('target1', )

        data_grad = self.vm.get_batch_input_gradient(perturbed_image, target_labels)
        print('Data grad:', data_grad)
        print(data_grad.shape)
        input()

        # ------------END TODO-------------

        

        return perturbed_image.cpu().detach().numpy()


