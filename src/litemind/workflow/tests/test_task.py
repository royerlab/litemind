import pytest

from litemind import OpenAIApi
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.workflow.task import Task


class DummyTask(Task):
    def build_message(self) -> Message:

        # If there is some dependency, it should be passed as an argument
        text_to_translate = self.dependencies["text"].get_result()

        # Create the message content
        content = f"Text to translate:\n```\n{text_to_translate}\n```\n"

        # Return the message
        return Message(content)

    def validate_result(self, result: str) -> bool:
        # Valid if the result starts with "Response:"
        return len(result.strip()) > 0


# pytest fictures for agents:
@pytest.fixture
def agent():
    api = OpenAIApi()

    # Create the agent:
    expert_agent = Agent(api=api)
    expert_agent.append_system_message(
        "You are an expert translator. You translate text to English."
    )

    return expert_agent


def test_store_and_load_result(tmp_path, agent):

    folder = tmp_path / "folder"
    folder.mkdir()
    task = DummyTask(name="test", agent=agent, folder=str(folder))
    result_text = "Response: dummy result"

    # Test storing result.
    task._store_result(result_text)
    file_path = folder / "test.md"
    assert file_path.exists()

    # Test loading result.
    loaded = task._load_result()
    assert loaded == result_text


def test_is_complete(tmp_path, agent):
    folder = tmp_path / "folder"
    folder.mkdir()
    task = DummyTask(name="complete", agent=agent, folder=str(folder))

    # Initially, file does not exist.
    assert not task.is_complete()

    # Store a valid result.
    valid_result = "Response: valid result"
    task._store_result(valid_result)
    assert task.is_complete()

    # Replace file content with an invalid result.
    file_path = folder / "complete.md"
    with open(file_path, "w") as f:
        f.write("")
    assert not task.is_complete()


def test_run_with_dependencies(tmp_path, agent):
    folder = tmp_path / "folder"
    folder.mkdir()

    # Create a dependency task and complete it.
    dep = DummyTask(name="text", agent=agent, folder=str(folder))
    dep._store_result("Bonjour le monde")  # French for "Hello world")

    # Create main task with the dependency.
    main = DummyTask(name="main", agent=agent, dependencies=[dep], folder=str(folder))
    result = main.run()
    assert "hello" in result.lower()


def test_run_without_completed_dependency(tmp_path, agent):
    folder = tmp_path / "folder"
    folder.mkdir()

    # Create a dependency task without completing it.
    dep = DummyTask(name="text", agent=None, folder=str(folder))
    main = DummyTask(name="main", agent=agent, dependencies=[dep], folder=str(folder))

    main.run()
