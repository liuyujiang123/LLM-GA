import re
import time
from ...llm.interface_LLM import InterfaceLLM


class Evolution():

    def __init__(self, api_endpoint, api_key, model_LLM, llm_use_local, llm_local_url, debug_mode, prompts, **kwargs):

        # set prompt interface
        # getprompts = GetPrompts()
        self.prompt_task = prompts.get_task()
        self.prompt_func_name = prompts.get_func_name()
        self.prompt_func_inputs = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf = prompts.get_inout_inf()
        self.prompt_other_inf = prompts.get_other_inf()

        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode  # close prompt checking

        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM, llm_use_local, llm_local_url,
                                          self.debug_mode)

    def get_prompt_i1(self):

        prompt_content = (self.prompt_task + "\n" +
                          "First, analyze the nature of gradient-based adversarial attacks and how they can be optimized. "
                          "Then, consider what issues exist with gradient-based attacks and how to address them. For example, "
                          "an attack may get stuck in a poor local optimum, which can be mitigated by using momentum. "
                          "Finally, based on the analysis, describe your new algorithm and the main steps."
                          "The description must be inside a brace. Please implement it in Python as a function named"
                          + self.prompt_func_name + ". This function should accept " + str(len(self.prompt_func_inputs))
                          + " input(s): " + self.joined_inputs + ". The function should return " + str(len(self.prompt_func_outputs))
                          + " output(s): " + self.joined_outputs + ". " + self.prompt_inout_inf + " "+ self.prompt_other_inf
                          + "Do not give additional explanations.")
        return prompt_content

    def get_prompt_e1(self, indivs, bad_ideas):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = prompt_indiv + "No." + str(i + 1) + " algorithm and the corresponding code are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i]['code'] + "\n"

        bad_indivs = ""
        for i in range(len(bad_ideas)):
            bad_indivs = bad_indivs + "No." + str(i + 1) + " The algorithm is: \n" + bad_ideas[i]

        prompt_content = (self.prompt_task + "\n"
                          "I have " + str(len(indivs)) + " existing algorithms with their codes as follows: \n"
                          + prompt_indiv +
                          "Please help me create a new algorithm that has a totally different form from the given ones.\n"
                          "First, analyze the nature of gradient-based adversarial attacks and understand the implementation logic of the given algorithm "
                          "Then, analyze the factors that may affect the performance of gradient-based attacks, and consider "
                          "which methods could enhance the effectiveness of gradient-based attacks or better acquire gradients."
                          "Finally, based on the analysis, describe your new algorithm and the main steps."
                          "The description must be inside a brace. Please implement it in Python as a function named"
                          + self.prompt_func_name + ". This function should accept " + str(len(self.prompt_func_inputs))
                          + " input(s): " + self.joined_inputs + ". The function should return " + str(len(self.prompt_func_outputs))
                          + " output(s): " + self.joined_outputs + ". " + self.prompt_inout_inf + " " + self.prompt_other_inf + "\n"
                          + "Do not give additional explanations. \n"
                          + "Here are evaluated solutions that were rejected: \n"
                          + bad_indivs +
                          "When generating the algorithm, please avoid making the same mistakes you did with the evaluated rejected solutions.")
        return prompt_content

    def get_prompt_e2(self, indivs, bad_ideas):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = prompt_indiv + "No." + str(i + 1) + " algorithm and the corresponding code are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i]['code'] + "\n"

        bad_indivs = ""
        for i in range(len(bad_ideas)):
            bad_indivs = bad_indivs + "No." + str(i + 1) + " The algorithm is: \n" + bad_ideas[i]

        prompt_content = (self.prompt_task + "\n"
                          + "I have " + str(len(indivs)) + " existing algorithms with their codes as follows: \n"
                          + prompt_indiv + "\n"
                          "Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them. \n"
                          "First, identify the core method used in the given algorithms and their functions. "
                          "Secondly, by comparing these algorithms, identify the common characteristics they share as well as their unique approaches."
                          "Thirdly, based on the above analysis, select one or more aspects as the foundation for innovation in a new algorithm, "
                          "such as integrating the core methods of the aforementioned algorithms, introducing new methods, or altering the approach to solving the problem. "
                          "Finally, based on the analysis, describe your new algorithm and the main steps."
                          "The description must be inside a brace. Please implement it in Python as a function named"
                          + self.prompt_func_name + ". This function should accept " + str(len(self.prompt_func_inputs))
                          + " input(s): " + self.joined_inputs + ". The function should return " + str(len(self.prompt_func_outputs))
                          + " output(s): " + self.joined_outputs + ". " + self.prompt_inout_inf + " " + self.prompt_other_inf
                          + "Do not give additional explanations.\n"
                          + "Here are evaluated solutions that were rejected: \n"
                          + bad_indivs +
                          "When generating the algorithm, please avoid making the same mistakes you did with the evaluated rejected solutions.")


        return prompt_content

    def get_prompt_m1(self, indiv1, bad_ideas):
        bad_indivs = ""
        for i in range(len(bad_ideas)):
            bad_indivs = bad_indivs + "No." + str(i + 1) + " The algorithm is: \n" + bad_ideas[i]

        prompt_content = (self.prompt_task + "\n"
                          + "I have one algorithm with its code as follows:\n."
                          "Algorithm description: " + indiv1['algorithm'] + '\n'
                          "Code:\n" + indiv1['code'] + "\n"
                          "Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided. \n"
                          "Firstly, carefully read and understand the functionality of the provided algorithm as well as its implementation logic. "
                          "Secondly, identify the key components and core ideas of the algorithm, and analyze the strengths and limitations of the algorithm."
                          "Thirdly, consider how to introduce innovation while maintaining the functionality of the algorithm, and think about which aspects can be optimized."
                          "Finally, based on the above analysis, describe your new algorithm and the main steps."
                          "The description must be inside a brace. Next, implement it in Python as a function named"
                          + self.prompt_func_name + ". This function should accept " + str(len(self.prompt_func_inputs))
                          + " input(s): " + self.joined_inputs + ". The function should return " + str(len(self.prompt_func_outputs))
                          + " output(s): " + self.joined_outputs + ". " + self.prompt_inout_inf + " " + self.prompt_other_inf
                          + "Do not give additional explanations.\n"
                          + "Here are evaluated solutions that were rejected: \n"
                          + bad_indivs +
                          "When generating the algorithm, please avoid making the same mistakes you did with the evaluated rejected solutions.")
        return prompt_content

    def get_prompt_m2(self, indiv1, bad_ideas):
        bad_indivs = ""
        for i in range(len(bad_ideas)):
            bad_indivs = bad_indivs + "No." + str(i + 1) + " The algorithm is: \n" + bad_ideas[i]

        prompt_content = (self.prompt_task + "\n"
                          "I have one algorithm with its code as follows\n."
                          "Algorithm description: " + indiv1['algorithm'] + "\n"
                          "Code:\n" + indiv1['code'] + "\n"
                          "Please identify the main algorithm parameters and assist me in creating a new algorithm that has "
                          "a different parameter settings of the score function provided. \n"
                          "Firstly, understand the implementation logic of the given algorithm, and identify the key components and core ideas of the algorithm. "
                          "Secondly, identify the key parameters in the algorithm, and analyze their roles as well as their impact on the algorithm's performance."
                          "Thirdly, based on the above analysis, decide which parameters need to be adjusted and how to adjust them. "
                          "Determine the new parameter settings to ensure they can improve the algorithm's performance."
                          "Finally, describe your new algorithm and the main steps. The description must be inside a brace. "
                          "Please implement it in Python as a function named" + self.prompt_func_name + ". This function should accept "
                          + str(len(self.prompt_func_inputs)) + " input(s): " + self.joined_inputs + ". The function should return "
                          + str(len(self.prompt_func_outputs)) + " output(s): " + self.joined_outputs + ". " + self.prompt_inout_inf
                          + " " + self.prompt_other_inf + "\n" + "Do not give additional explanations.\n"
                          + "Here are evaluated solutions that were rejected: \n"
                          + bad_indivs +
                          "When generating the algorithm, please avoid making the same mistakes you did with the evaluated rejected solutions.")
        return prompt_content

    def _get_alg(self, prompt_content):

        response = self.interface_llm.get_response(prompt_content)

        algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        if len(algorithm) == 0:
            if 'python' in response:
                algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
            elif 'import' in response:
                algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
            else:
                algorithm = re.findall(r'^.*?(?=def)', response, re.DOTALL)

        code = re.findall(r"import.*return", response, re.DOTALL)
        if len(code) == 0:
            code = re.findall(r"def.*return", response, re.DOTALL)

        n_retry = 1
        while (len(algorithm) == 0 or len(code) == 0):
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")

            response = self.interface_llm.get_response(prompt_content)

            algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
            if len(algorithm) == 0:
                if 'python' in response:
                    algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
                elif 'import' in response:
                    algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
                else:
                    algorithm = re.findall(r'^.*?(?=def)', response, re.DOTALL)

            code = re.findall(r"import.*return", response, re.DOTALL)
            if len(code) == 0:
                code = re.findall(r"def.*return", response, re.DOTALL)

            if n_retry > 3:
                break
            n_retry += 1

        algorithm = algorithm[0]
        code = code[0]

        code_all = code + " " + ", ".join(s for s in self.prompt_func_outputs)

        return [code_all, algorithm]

    def i1(self):

        prompt_content = self.get_prompt_i1()

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def e1(self, parents, bad_ideas):

        prompt_content = self.get_prompt_e1(parents, bad_ideas)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def e2(self, parents, bad_ideas):

        prompt_content = self.get_prompt_e2(parents, bad_ideas)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def m1(self, parents, bad_ideas):

        prompt_content = self.get_prompt_m1(parents, bad_ideas)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def m2(self, parents, bad_ideas):

        prompt_content = self.get_prompt_m2(parents, bad_ideas)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
