"""
Task abstraction for agent-driven workflows.

Provides the :class:`Task` abstract base class, which represents a single
unit of work executed by an :class:`~litemind.agent.agent.Agent`.  Tasks can
declare dependencies on other tasks, forming a directed acyclic graph (DAG).
Results and prompts are persisted to disk so that completed tasks are not
re-executed on subsequent runs.
"""

import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Optional

from arbol import aprint, asection

from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message


class Task(ABC):
    """Abstract base class representing a single unit of work in a workflow.

    Tasks can be organized into a directed acyclic graph (DAG) via
    dependencies, forming a workflow.  Subclasses must implement
    :meth:`build_message` (to create the prompt sent to the agent) and
    :meth:`validate_result` (to verify that the agent's response is
    acceptable).

    Each task persists its prompt and result to disk so that completed
    tasks are not re-executed on subsequent runs.
    """

    def __init__(
        self,
        name: str,
        agent: Optional[Agent] = None,
        dependencies: Optional[List["Task"]] = None,
        folder: Optional[str] = None,
        save_pdf: bool = False,
    ):
        """
        Initialize the task.

        Parameters
        ----------
        name : str
            A unique name for this task. Used as the base filename for
            persisted prompts and results.
        agent : Agent, optional
            The agent that will execute this task, by default ``None``.
        dependencies : list of Task, optional
            Tasks whose results this task depends on, by default ``None``.
        folder : str, optional
            Directory in which to store prompt and result files. Defaults
            to the current working directory.
        save_pdf : bool, optional
            Whether to also save the result as a PDF file, by default
            ``False``.
        """

        # Basic attributes:
        self.name = name
        self.agent = agent

        # Dependencies:
        self.dependencies: dict[str, Task] = {}

        # Folder to store results:
        self.folder = folder if folder is not None else "."

        # Make sure the folder exists:
        os.makedirs(self.folder, exist_ok=True)

        # Add provided dependencies:
        for dep in dependencies or []:
            self.add_dependency(dep)

        self.save_pdf = save_pdf

    def add_dependency(self, task: "Task"):
        """Register another task as a dependency of this one.

        Parameters
        ----------
        task : Task
            The task that must complete before this task can run.
        """
        self.dependencies[task.name] = task

    @abstractmethod
    def build_message(self) -> Message:
        """Build the prompt message to send to the agent.

        Subclasses must implement this method.  The message may reference
        results from dependency tasks (available via ``self.dependencies``)
        or any attributes set on the task instance.

        Returns
        -------
        Message
            The message containing the prompt for the agent.
        """
        pass

    @abstractmethod
    def validate_result(self, result: str) -> bool:
        """Check whether the agent's result is acceptable.

        Subclasses must implement this method.  Validation may inspect
        content, length, or any other criteria (e.g., checking for
        common LLM artifacts).

        Parameters
        ----------
        result : str
            The result string returned by the agent.

        Returns
        -------
        bool
            ``True`` if the result passes validation, ``False`` otherwise.
        """
        return True

    def post_process_result_before_saving_pdf(self, result: str) -> str:
        """Post-process the result string before it is converted to PDF.

        Override this method to reformat the markdown, strip artifacts, or
        add extra content before PDF generation.

        Parameters
        ----------
        result : str
            The raw result markdown string.

        Returns
        -------
        str
            The processed markdown string to be rendered as PDF.
        """
        return result

    def __call__(self) -> str:
        """Execute the task by delegating to :meth:`run`.

        Returns
        -------
        str or None
            The task result, or ``None`` if the task could not be run.
        """
        return self.run()

    def run(self) -> str:
        """Execute the task and return its result.

        If any dependency tasks have not been completed, they are run
        first.  If this task has already been completed (a valid result
        file exists on disk), the cached result is returned without
        re-executing.

        Returns
        -------
        str or None
            The task result as a markdown string, or ``None`` if the
            task could not be executed (e.g., no agent assigned or an
            error occurred).
        """

        with asection(f"Running task: {self.name}"):

            # Check that all dependencies are complete and run them if not:
            for name, task in self.dependencies.items():
                if not task.is_complete():
                    aprint(f"Dependency task '{name}' is not complete. Running it now.")
                    task.run()

            if self.agent is not None:

                # Build the message:
                message = self.build_message()

                # Store the prompt:
                self._store_prompt(message.to_markdown())

                # Check if the task is already complete:
                # We do this 'last minute' so that we can still store the prompt
                if self.is_complete():
                    aprint(
                        f"Task '{self.name}' is already complete. Skipping execution."
                    )
                    result = self._load_result()
                else:
                    try:
                        # Send the message to the agent:
                        response = self.agent(message)

                        # Get the result as a string:
                        result = response[-1].to_markdown()

                        # Store the result:
                        self._store_result(result)

                        aprint(
                            f"Task '{self.name}' completed. Result stored in '{self._get_result_file_path()}'."
                        )
                    except Exception as e:
                        # print stack trace for debugging:
                        import traceback

                        traceback.print_exc()

                        # Handle any exceptions that occur during the agent's execution:
                        aprint(
                            f"An error occurred while running the task '{self.name}': {e}"
                        )
                        result = None

            else:
                aprint(f"No agent assigned to task '{self.name}'. Cannot run the task.")
                result = None

            if result is not None and self.save_pdf:
                self.save_as_pdf(result)

            return result

    def is_complete(self) -> bool:
        """Check whether this task has already been completed.

        A task is considered complete when its result file exists on disk
        and the stored result passes :meth:`validate_result`.

        Returns
        -------
        bool
            ``True`` if a valid result is already persisted, ``False``
            otherwise.
        """

        # Check if the result file exists:
        file_exists = os.path.exists(self._get_result_file_path())

        # Read the result file and check if it is valid:
        if file_exists:
            result = self._load_result()
            if self.validate_result(result):
                aprint(f"Task '{self.name}' is complete.")
                return True
            else:
                aprint(f"Task '{self.name}' is not complete: result validation failed.")
                return False
        else:
            aprint(f"Task '{self.name}' is not complete: result file does not exist.")
            return False

    def get_prompt(self) -> Optional[str]:
        """Return the stored prompt for this task, if available.

        Returns
        -------
        str or None
            The prompt markdown string, or ``None`` if no prompt file
            has been written yet.
        """
        if os.path.exists(self._get_prompt_file_path()):
            return self._load_prompt()
        else:
            aprint(f"No prompt found for task '{self.name}'.")
            return None

    def get_result(self) -> Optional[str]:
        """Return the stored result for this task, if available.

        Returns
        -------
        str or None
            The result markdown string, or ``None`` if the task has not
            been completed yet.
        """
        if os.path.exists(self._get_result_file_path()):
            return self._load_result()
        else:
            aprint(f"No result found for task '{self.name}'.")
            return None

    def save_as_pdf(self, result: Optional[str]):
        """Save the task output as a PDF file in the task's folder.

        The *result* is first passed through
        :meth:`post_process_result_before_saving_pdf` before rendering.
        If *result* is ``None``, the stored prompt is used instead.

        Parameters
        ----------
        result : str or None
            The markdown content to render.  If ``None``, the previously
            stored prompt is loaded from disk.
        """
        try:
            # Get the PDF file path:
            pdf_file_path = self._get_pdf_file_path()

            # Check if the file already exists and delete it if it does:
            if os.path.exists(pdf_file_path):
                os.remove(pdf_file_path)
                aprint(f"Existing PDF file '{pdf_file_path}' deleted.")

            # Load the prompt and post-process it before saving:
            markdown = self.post_process_result_before_saving_pdf(
                result or self._load_prompt()
            )

            # Save the markdown content as a PDF:
            from md2pdf import md2pdf

            md2pdf(pdf_file_path=pdf_file_path, md_content=markdown)
            aprint(f"Task '{self.name}' saved as PDF in '{pdf_file_path}'.")
        except FileNotFoundError:
            aprint(f"Prompt file not found for task '{self.name}'. Cannot save as PDF.")
            return
        except Exception as e:
            aprint(f"An error occurred while saving the task '{self.name}' as PDF: {e}")
            return

    def _store_prompt(self, prompt: str):
        """Persist the prompt markdown to disk.

        Overwrites any existing prompt file and invalidates the
        :meth:`_load_prompt` cache so subsequent reads return fresh data.

        Parameters
        ----------
        prompt : str
            The prompt markdown string to write.
        """
        # Clear the cache since we're updating the prompt file
        self._load_prompt.cache_clear()
        # if the file exists already, then delete it:
        if os.path.exists(self._get_prompt_file_path()):
            os.remove(self._get_prompt_file_path())
        # save the prompt string in a file called <task_name>.prompt.md in the folder:
        with open(self._get_prompt_file_path(), "w") as f:
            f.write(prompt)
        aprint(f"Prompt stored in '{self._get_prompt_file_path()}'.")

    @lru_cache(maxsize=None)
    def _load_prompt(self) -> str:
        """Load the prompt markdown from disk.

        Results are cached via :func:`functools.lru_cache` so that repeated
        reads within the same run do not hit the filesystem.

        Returns
        -------
        str
            The prompt markdown string read from the prompt file.
        """
        # load the prompt string from a file called <task_name>.prompt.md in the folder:
        with open(self._get_prompt_file_path(), "r") as f:
            prompt = f.read()
        aprint(f"Prompt loaded from '{self._get_prompt_file_path()}'.")
        return prompt

    def _store_result(self, result: str):
        """Persist the task result to disk.

        Overwrites any existing result file.

        Parameters
        ----------
        result : str
            The result markdown string to write.
        """
        # if file exists already then delete it:
        if os.path.exists(self._get_result_file_path()):
            os.remove(self._get_result_file_path())
        # save the result string in a file called <task_name>.txt in the folder:
        with open(self._get_result_file_path(), "w") as f:
            f.write(result)
        aprint(f"Result stored in '{self._get_result_file_path()}'.")

    def _load_result(self) -> str:
        """Load the task result from disk.

        Returns
        -------
        str
            The result markdown string read from the result file.
        """
        # load the result string from a file called <task_name>.txt in the folder:
        with open(self._get_result_file_path(), "r") as f:
            result = f.read()
        aprint(f"Result loaded from '{self._get_result_file_path()}'.")
        return result

    def _get_prompt_file_path(self) -> str:
        """Return the filesystem path for this task's prompt file.

        Returns
        -------
        str
            Path of the form ``<folder>/<name>.prompt.md``.
        """
        return os.path.join(self.folder, f"{self.name}.prompt.md")

    def _get_result_file_path(self) -> str:
        """Return the filesystem path for this task's result file.

        Returns
        -------
        str
            Path of the form ``<folder>/<name>.md``.
        """
        return os.path.join(self.folder, f"{self.name}.md")

    def _get_pdf_file_path(self) -> str:
        """Return the filesystem path for this task's PDF output file.

        Returns
        -------
        str
            Path of the form ``<folder>/<name>.pdf``.
        """
        return os.path.join(self.folder, f"{self.name}.pdf")

    def get_folder_name(self) -> str:
        """Return the leaf directory name of this task's output folder.

        Returns
        -------
        str
            The last component of the folder path (e.g. ``"results"``
            for ``"/tmp/results"``).
        """
        return os.path.basename(self.folder)

    def __repr__(self) -> str:
        """Return a developer-friendly string representation of the task.

        Returns
        -------
        str
            A string showing the task name, assigned agent, and
            dependency mapping.
        """
        return f"Task(name={self.name}, agent={self.agent}, dependencies={self.dependencies})"
