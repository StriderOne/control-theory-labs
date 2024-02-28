from abc import ABC, abstractmethod
from  LatexGapFiller import LatexGapFiller, TextSheet
from ControlTheoryToolbox import System, CToolbox
import json
import matplotlib.pyplot as plt
import scipy
import sympy
import numpy as np
import control 

class Task(ABC):

    def __init__(self, id, config=None, latex_gap_filler: LatexGapFiller = None):
        self._id = id
        self._config = config
        self._sheet = TextSheet(config["sheet"])
        self._latex_gap_filler = latex_gap_filler

    @abstractmethod
    def complete_task(self):
        pass

class ControlTheoryTaskManager():
    
    def __init__(self, config :dict) -> None:
        self.lab_number = config['lab']['number'] 
        self.variant = config['lab']['variant']
        self.tasks_configs = config['lab']['tasks']
        self.latex_gap_filler = LatexGapFiller()
        self.tasks = []

    def create_task(self, task):
        self.tasks.append(task(len(self.tasks), self.tasks_configs[len(self.tasks)], self.latex_gap_filler))

    def complete_tasks(self):
        for task in self.tasks:
            task.complete_task()

    def generate_report(self):
        self.latex_gap_filler.generate_text()

def modeling(A, B, K, image_name):
    ss = control.ss((A + B @ K), B*0, A * 0, B * 0)

    time = np.linspace(0, 3, 1000)
    output = control.forced_response(ss, X0=[1, 1, 1, 1], T=time).states

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4,hspace=0.4)
    for i in range(4):
        axs[i//2, i % 2].plot(time, output[i], linewidth=4)
        # axs[i//2, i % 2].scatter(t[-1], x1[i], color='r', linewidth=4, label='goal')
        axs[i//2, i % 2].set_title(f'x_{i} = x_{i}(t)', fontsize=12)
        axs[i//2, i % 2].set_xlabel(f"t, [c]", fontsize=12)
        axs[i//2, i % 2].grid(True)
        axs[i//2, i % 2].legend()
    
    plt.savefig(f'{image_name}.png')
        
    # axs[1, 1].plot(t, np.array(U).reshape(-1), linewidth=4)
    # axs[1, 1].set_title(f'u = u(t)', fontsize=12)
    # axs[1, 1].set_xlabel(f"t, [c]", fontsize=12)
    # axs[1, 1].grid(True)

class Task1(Task):

    def __init__(self, id, config, latex_gap_filler:LatexGapFiller):
        super().__init__(id, config, latex_gap_filler)

    def complete_task(self):
        config = self._config
        system = System(config["A"], config["B"], 0, 0)
        print(self._sheet.key_words)
        eigen_values = np.linalg.eig(system.A)[0]
        self._latex_gap_filler.add_key_word("A", system.A)
        self._latex_gap_filler.add_key_word("B", system.B)
        self._latex_gap_filler.add_key_word("lambdas", eigen_values)
        controllable_eigen_values = eigen_values[np.where(CToolbox.check_eigenvalues_controllable(system.A, system.B))]
        not_controllable_eigen_values = eigen_values[np.where(CToolbox.check_eigenvalues_controllable(system.A, system.B) == False)]
        self._latex_gap_filler.add_key_word("controllable_eigen_values", controllable_eigen_values)
        self._latex_gap_filler.add_key_word("not_controllable_eigen_values", not_controllable_eigen_values)

        for i, desired_eigvalues_raw in enumerate(self._config["eigen_values"]):
            desired_eigvalues_raw = np.array(desired_eigvalues_raw)
            desired_eigvalues = np.zeros(desired_eigvalues_raw.shape[0], complex)
            desired_eigvalues.real = desired_eigvalues_raw[:, 0]
            desired_eigvalues.imag = desired_eigvalues_raw[:, 1]
            self._latex_gap_filler.add_key_word("desired_eigen_values", desired_eigvalues)
            G = np.diag(desired_eigvalues)
            if i == 0:
                G[0, 1] = 1
                G[1, 2] = 1
                G[2, 3] = 1

            self._latex_gap_filler.add_key_word("G", G)

            Y = np.array([[0, 1, 0, 0]])
            self._latex_gap_filler.add_key_word("Y", Y)
            P = scipy.linalg.solve_sylvester(system.A, -G, system.B@Y)
            # print(P)
            self._latex_gap_filler.add_key_word("P", P)
            K = -Y @ np.linalg.pinv(P)
            print(K)
            self._latex_gap_filler.add_key_word("K", K)

            self._latex_gap_filler.add_key_word("id", "1." + str(i))
            self._latex_gap_filler.add_key_word("image_name", "image1." + str(i) + ".png")
            modeling(system.A, system.B, K, "image1." + str(i))
            self._latex_gap_filler.add_sheet(self._sheet)
            

if __name__ == '__main__':
    config_file = open('./lab8/lab.json')

    config = json.load(config_file)

    task_manager = ControlTheoryTaskManager(config)
    task_manager.create_task(Task1)
    task_manager.complete_tasks()

    task_manager.generate_report()


