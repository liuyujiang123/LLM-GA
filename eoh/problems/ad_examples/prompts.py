class GetPrompts:
    def __init__(self):
        self.prompt_task = ("I am researching the security of image classification models and would like "
                            "to implement a gradient-based attack method to generate adversarial examples. "
                            "My setup is as follows:"
                            "1. A batch of multi-label original images (org_img)."
                            "2. A multi-label classification model based on PyTorch (model)."
                            "I want to find a method that can apply subtle perturbations to these original"
                            "images, thereby generating adversarial examples (adv_img), such that when these"
                            "adversarial examples are input into the model, the model will classify them as the"
                            "specified target label (target). Please help me design a gradient-based attack" 
                            "algorithm that should meet the following requirements:"
                            "1. Implemented using the PyTorch framework."
                            "2. Gradually adjusts the images by computing the gradients of the loss function with "
                            "respect to the input images until the model's output approaches the target label."
                            "3. The added perturbation should be as small as possible.")
        self.prompt_func_name = "gen_adv_examples"
        self.prompt_func_inputs = ["org_img", "model", "target"]
        self.prompt_func_outputs = ["adv_img"]
        self.prompt_inout_inf = ("org_img and adv_img and target are Tensor, \
                                 model is a deep neural network implemented using PyTorch")
        self.prompt_other_inf = ("I have already written the class framework, which includes the helper functions and "
                                 "the necessary parameters. Therefore, you only need to focus on generating the "
                                 "gen_adv_examples(self, org_img, model, target) function. Please strictly follow the following "
                                 "framework to generate the gen_adv_examples function, and do not generate any classes or "
                                 "additional helper functions." + basic_algorithm + '\n' +
                                 "Please generate the gen_adv_examples function, adhering to the above framework, and use the following assumptions:\n"
                                 + "1. self.get_gradient(adv_img, model, target) is already defined and used to obtain the gradient.\n"
                                 + "2. self.clip_adv(org_img, adv_img) is already defined and used to clip the adversarial examples.\n"
                                 + "3. The parameters maxiter, epsilon, alpha, and the loss function are already defined in the class.\n"
                                 + "4. You can directly use these parameters and functions without re-implementing them.\n"
                                 + "5. The class only contains the parameters and functions mentioned above. "
                                   "If you need to use other parameters, you must create them first.")

    def get_task(self):
        return self.prompt_task

    def get_func_name(self):
        return self.prompt_func_name

    def get_func_inputs(self):
        return self.prompt_func_inputs

    def get_func_outputs(self):
        return self.prompt_func_outputs

    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf

    def get_other_inf2(self):
        return self.prompt_other_inf2

basic_algorithm = "Some basic gradient-based attack algorithm implementations are as follows \n:\
import torch\
import Variable\
def i_fgsm(self, org_img, model, target):\
    adv_img = Variable(org_img.data, requires_grad=True)\
    for _ in range(self.maxiter):\
        grad = self.get_gradient(adv_img, model, target)\
        adv_img = adv_img + self.alpha * grad.sign()\
        adv_img = self.clip_adv(org_img, adv_img)\
    return adv_img\n \
import torch\
import Variable\
def mi_fgsm(self, org_img, model, target):\
    g = 0\
    decay_factor = 1.0\
    adv_img = Variable(org_img.data, requires_grad=True)\
    for _ in range(self.maxiter):\
        grad = self.get_gradient(adv_img, model, target)\
        g = decay_factor * g + grad / torch.norm(adv_img.grad, p=1, dim=[1, 2, 3], keepdim=True)\
        adv_img = adv_img + self.alpha * g.sign()\
        adv_img = self.clip_adv(org_img, adv_img)\
    return adv_img"
# import torch\
# import Variable\
# def PGD(self, org_img, model, target):\
#     adv_img = org_img + torch.empty_like(org_img).uniform_(-self.epsilon, self.epsilon)\
#     adv_img = Variable(adv_img.data, requires_grad=True)\
#     for _ in range(self.maxiter):\
#         grad = self.get_gradient(adv_img, model, target)\
#         adv_img = adv_img + self.alpha * adv_img.grad.sign()\
#         adv_img = self.clip_adv(org_img, adv_img)\
#     return adv_img"