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


    def attack(self, original_image:  np.ndarray, labels: List[int], target_label: int, epsilon = 0.3):
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
        # To run this file change the group_name(#L:41) in attack_project-Evaluation.py to "FGSM_Targeted"

        alpha = 2/255
        perturbed_image = torch.FloatTensor(perturbed_image)

        perturbed_image = perturbed_image + torch.empty_like(perturbed_image).uniform_(-epsilon, epsilon)
        perturbed_image = torch.clamp(perturbed_image, min=0, max=1).detach()

        iters = 40 # int(min(epsilon + 4, 1.25*epsilon))

        original_tensor = torch.FloatTensor(original_image)

        target_labels = [target_label]*len(labels)

        loss = torch.nn.CrossEntropyLoss()

        for i in range(iters):

            # # Calculate the gradient of loss with respect to the image and target_label
            # data_grad = self.vm.get_batch_input_gradient(perturbed_image, target_labels)
            # data_grad = torch.FloatTensor(data_grad)

            # # Determine the direction of gradient using sign()
            # sign_data_grad = data_grad.sign()

            # # Perturb the image in the direction of gradient with respect to target_label by epsilon
            # perturbed_image = torch.FloatTensor(perturbed_image) + alpha * sign_data_grad
            
            # delta = torch.clamp(perturbed_image - original_image, min=-epsilon, max=epsilon)
            # perturbed_image = torch.clamp(original_tensor + delta, min=0, max=1).detach()

            # Clamp the value of each pixel to be between 0 & 1
            # perturbed_image = torch.clamp(perturbed_image, 0, 1)


            output = torch.FloatTensor(self.vm.get_batch_output(perturbed_image))
            # Calculate loss
            cost = -loss(output, torch.LongTensor(target_labels))

            # print(cost)
            # input()

            # Update adversarial images
            grad = torch.autograd.grad(cost, perturbed_image,
                                       retain_graph=False, create_graph=False)[0]

            perturbed_image = perturbed_image.detach() + alpha*grad.sign()
            delta = torch.clamp(perturbed_image - original_tensor, min=-epsilon, max=epsilon)
            perturbed_image = torch.clamp(original_tensor + delta, min=0, max=1).detach()

            

        # ------------END TODO-------------

        

        return perturbed_image.cpu().detach().numpy()


