from typing import Dict, Type, TypeVar

from agents.base.agent import AgentBase
from exp.base.base import ExperimentBase

experiments: Dict[str, Type] = {}
agents: Dict[str, Type] = {}

Experiment = TypeVar("Experiment", bound=ExperimentBase)
Agent = TypeVar("Agent", bound=AgentBase)


def register_experiment(name: str):
    def decorator(cls: Type[Experiment]) -> Type[Experiment]:
        experiments[name] = cls
        return cls

    return decorator


def register_agent(name: str):
    def decorator(cls: Type[Agent]) -> Type[Agent]:
        agents[name] = cls
        return cls

    return decorator


def get_experiment(name: str) -> Type[Experiment]:
    return experiments.get(name)


def get_agent(name: str) -> Type[Agent]:
    return agents.get(name)
