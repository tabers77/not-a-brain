from not_a_brain.tasks.base import TaskBase
from not_a_brain.tasks.synthetic.arithmetic import ArithmeticTask
from not_a_brain.tasks.synthetic.copy_task import CopyTask
from not_a_brain.tasks.synthetic.grammar import GrammarTask
from not_a_brain.tasks.synthetic.knowledge_qa import KnowledgeQATask
from not_a_brain.tasks.synthetic.compositional import CompositionalTask
from not_a_brain.tasks.synthetic.unknown import UnknownTask

ALL_TASKS = {
    "arithmetic": ArithmeticTask,
    "copy": CopyTask,
    "grammar": GrammarTask,
    "knowledge_qa": KnowledgeQATask,
    "compositional": CompositionalTask,
    "unknown": UnknownTask,
}

__all__ = [
    "TaskBase",
    "ArithmeticTask",
    "CopyTask",
    "GrammarTask",
    "KnowledgeQATask",
    "CompositionalTask",
    "UnknownTask",
    "ALL_TASKS",
]
