import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Optional

from arbol import aprint, asection

from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message


class Task(ABC):
    """
    Abstract class that represents a task.
    Tasks can be combined into a workflow -- a graph of tasks in which tasks can depend on other tasks.
    Users must implement classes that derive from this base class and implement both the build_message()
    and validate_result() methods.

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
        Initializes the Task.
        Ideally, you want to customize the constructor of your derived class.

        Parameters
        ----------
        name : str
            The name of the task
        agent : Agent
            The agent to execute the task
        dependencies : Optional[List['Task']], optional
            The list of dependencies (results of other tasks), by default None
        save_pdf: bool
            Whether to save the task's result as a PDF file, by default False
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
        """
        Adds a dependency to this task.

        Parameters
        ----------
        task: Task
            The task to add as a dependency.

        """
        self.dependencies[task.name] = task

    @abstractmethod
    def build_message(self) -> Message:
        """
        Build the message to send to the agent.
        This method should be implemented by the user to create a message
        that contains the necessary information for the agent to execute the task.
        The message should be a Message object that contains the prompt for the task.
        The prompt can depend on the results of other tasks, which can be accessed
        through the dependencies' dictionary, or can depend on the task's own attributes,
        which may have been set in the constructor.

        Returns
        -------
        Message
            The message to send to the agent.

        """
        pass

    @abstractmethod
    def validate_result(self, result: str) -> bool:
        """
        Validate the result of the task.
        This method should be implemented by the user to check if the result
        of the task is valid.
        The result is a string that is returned by the agent after executing the task.
        The validation can be based on the content of the result, its length, or any other criteria.
        Checking for typical 'LLM' artifacts or mistakes is a good example of validation.
        If the result is valid, the method should return True, otherwise False.

        Parameters
        ----------
        result: str
            The result of the task to validate.

        Returns
        -------
        bool
            True if the result is valid, False otherwise.

        """
        return True

    def post_process_result_before_saving_pdf(self, result: str) -> str:
        """
        Post-process the result before saving it to PDF.
        This can be used to format the result, add additional information, etc.
        """
        return result

    def __call__(self) -> str:
        """
        Allow calling the task as a function.
        This method will check if the task is complete, and if not, it will run the task.
        If dependencies are not complete, it will run them first.
        If the task is already complete, it will return the result without running the task again.
        If the task has an agent assigned, it will build the message and send it to the agent.
        If the task does not have an agent assigned, it will print a message and return None.
        If the task is complete, it will return the result.

        Returns
        -------
        str
            The result of the task, or None if the task could not be run.

        """
        return self.run()

    def run(self) -> str:

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
        """
        Check if the task is complete.

        Returns
        -------
        bool
            True if the task is complete, False otherwise.

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
        """
        Returns the prompt for the task, if it exists.
        If the prompt file does not exist, returns None.
        """
        if os.path.exists(self._get_prompt_file_path()):
            return self._load_prompt()
        else:
            aprint(f"No prompt found for task '{self.name}'.")
            return None

    def get_result(self) -> Optional[str]:
        """
        Returns the result for the task, if it exists.
        If the result file does not exist, returns None.
        """
        if os.path.exists(self._get_result_file_path()):
            return self._load_result()
        else:
            aprint(f"No result found for task '{self.name}'.")
            return None

    def save_as_pdf(self, result: Optional[str]):
        """
        Saves the task's prompt and result as a PDF file.
        The PDF will be saved in the specified folder or in the task's folder if no folder is specified.
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
        # load the prompt string from a file called <task_name>.prompt.md in the folder:
        with open(self._get_prompt_file_path(), "r") as f:
            prompt = f.read()
        aprint(f"Prompt loaded from '{self._get_prompt_file_path()}'.")
        return prompt

    def _store_result(self, result: str):
        # if file exists already then delete it:
        if os.path.exists(self._get_result_file_path()):
            os.remove(self._get_result_file_path())
        # save the result string in a file called <task_name>.txt in the folder:
        with open(self._get_result_file_path(), "w") as f:
            f.write(result)
        aprint(f"Result stored in '{self._get_result_file_path()}'.")

    def _load_result(self) -> str:
        # load the result string from a file called <task_name>.txt in the folder:
        with open(self._get_result_file_path(), "r") as f:
            result = f.read()
        aprint(f"Result loaded from '{self._get_result_file_path()}'.")
        return result

    def _get_prompt_file_path(self) -> str:
        return os.path.join(self.folder, f"{self.name}.prompt.md")

    def _get_result_file_path(self) -> str:
        return os.path.join(self.folder, f"{self.name}.md")

    def _get_pdf_file_path(self) -> str:
        return os.path.join(self.folder, f"{self.name}.pdf")

    def get_folder_name(self) -> str:
        """
        Returns the last part of the folder path
        """
        return os.path.basename(self.folder)

    def __repr__(self):
        return f"Task(name={self.name}, agent={self.agent}, dependencies={self.dependencies})"
