from abc import ABC, abstractmethod
from  LatexGapFiller import LatexGapFiller
from ControlTheoryToolbox import System, CToolbox
import json

class Task(ABC):

    def __init__(self, id, config=None):
        self._id = id
        self._config = config

    @abstractmethod
    def complete_task(self):
        pass

class ControlTheoryTaskManager():
    
    def __init__(self, config :dict) -> None:
        self.lab_number = config['lab']['number'] 
        self.variant = config['lab']['variant']
        self.tasks_configs = config['lab']['tasks']
        self.tasks = []

    def create_task(self, task):
        self.tasks.append(task(len(self.tasks), self.tasks_configs['task'+str(len(self.tasks))]))

    def complete_tasks(self):
        for task in self.tasks:
            task.complete_task()

    def generate_report(self):
        pass

class Task1(Task):

    def __init__(self, id, config):
        super().__init__(id, config)

    def complete_task(self):
        config = self._config
        system = System(config["A"], config["B"], 0, 0)

        


if __name__ == '__main__':
    config_file = open('./lab8/lab8.json')

    config = json.load(config_file)

    task_manager = ControlTheoryTaskManager(config)

    task_manager.complete_tasks()

    task_manager.generate_report()


