# Workflow Package

The `workflow` package provides an abstract Task framework for building multi-step workflows with dependency management, result caching, and optional PDF export.

## Package Structure

```
workflow/
├── __init__.py
├── task.py                  # Core Task abstraction
└── tests/
    └── test_task.py         # Task tests
```

## Core Components

### Task (`task.py`)

The `Task` class is an abstract base class for building workflow steps:

```python
from litemind.workflow.task import Task
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message

class ResearchTask(Task):
    def build_message(self) -> Message:
        message = Message(role="user")
        message.append_text("Research the topic: AI in healthcare")
        return message

    def validate_result(self, result: str) -> bool:
        # Check result quality
        return len(result) > 100 and "healthcare" in result.lower()

# Create and run task
agent = Agent(api=my_api)
task = ResearchTask(
    name="research",
    agent=agent,
    folder="/tmp/workflow",
    save_pdf=True
)

result = task.run()  # or task()
```

## Task Configuration

### Constructor Parameters

```python
Task(
    name: str,                           # Task identifier
    agent: Optional[Agent] = None,       # Agent to execute the task
    dependencies: Optional[List[Task]] = None,  # Dependent tasks
    folder: Optional[str] = None,        # Storage folder for results
    save_pdf: bool = False,              # Export results as PDF
)
```

### Abstract Methods

#### `build_message() -> Message`

Creates the prompt/message for the agent to process. Access dependency results here:

```python
def build_message(self) -> Message:
    # Access dependency results
    if "upstream" in self.dependencies:
        upstream_result = self.dependencies["upstream"].get_result()

    message = Message(role="user")
    message.append_text(f"Process this: {upstream_result}")
    return message
```

#### `validate_result(result: str) -> bool`

Validates the generated result. Return `False` to mark task as incomplete:

```python
def validate_result(self, result: str) -> bool:
    # Check for minimum length
    if len(result) < 100:
        return False

    # Check for required content
    if "error" in result.lower():
        return False

    # Check for LLM artifacts
    if "I cannot" in result or "As an AI" in result:
        return False

    return True
```

## Task Dependencies

Tasks can depend on other tasks, forming a directed acyclic graph (DAG):

```python
# Define tasks
research_task = ResearchTask(name="research", agent=agent)
analysis_task = AnalysisTask(name="analysis", agent=agent, dependencies=[research_task])
report_task = ReportTask(name="report", agent=agent, dependencies=[analysis_task])

# Running the final task automatically runs dependencies
result = report_task.run()
# Executes: research_task → analysis_task → report_task
```

### Adding Dependencies Dynamically

```python
task2.add_dependency(task1)
```

## Task State Management

### File Storage Pattern

Tasks store their state in the configured folder:

- **Prompts**: `{folder}/{name}.prompt.md`
- **Results**: `{folder}/{name}.md`
- **PDFs**: `{folder}/{name}.pdf`

### State Methods

```python
# Check if task is complete
task.is_complete()  # True if result exists and passes validation

# Get stored prompt
prompt = task.get_prompt()

# Get stored result
result = task.get_result()
```

## PDF Export

Enable PDF export for tasks:

```python
task = MyTask(name="report", agent=agent, save_pdf=True)
result = task.run()  # Automatically saves PDF

# Or manually save
task.save_as_pdf(result)
```

### Custom PDF Processing

Override `post_process_result_before_saving_pdf` to customize PDF content:

```python
def post_process_result_before_saving_pdf(self, result: str) -> str:
    # Add header
    return f"# Final Report\n\n{result}"
```

## Complete Example

```python
from litemind.workflow.task import Task
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind import CombinedApi

class DataCollectionTask(Task):
    def build_message(self) -> Message:
        msg = Message(role="user")
        msg.append_text("Collect data about machine learning trends in 2024.")
        return msg

    def validate_result(self, result: str) -> bool:
        return len(result) > 200

class AnalysisTask(Task):
    def build_message(self) -> Message:
        data = self.dependencies["data"].get_result()
        msg = Message(role="user")
        msg.append_text(f"Analyze the following data:\n\n{data}")
        return msg

    def validate_result(self, result: str) -> bool:
        return "analysis" in result.lower() or "finding" in result.lower()

class ReportTask(Task):
    def build_message(self) -> Message:
        analysis = self.dependencies["analysis"].get_result()
        msg = Message(role="user")
        msg.append_text(f"Write a report based on:\n\n{analysis}")
        return msg

# Create workflow
api = CombinedApi()
agent = Agent(api=api)

data_task = DataCollectionTask(name="data", agent=agent, folder="/tmp/workflow")
analysis_task = AnalysisTask(name="analysis", agent=agent, folder="/tmp/workflow")
analysis_task.add_dependency(data_task)

report_task = ReportTask(name="report", agent=agent, folder="/tmp/workflow", save_pdf=True)
report_task.add_dependency(analysis_task)

# Run entire workflow
final_report = report_task.run()
```

## Features

- **Automatic Dependency Resolution**: Running a task automatically runs incomplete dependencies
- **Result Caching**: Results are stored to disk and reused if valid
- **Validation**: Custom validation ensures result quality
- **PDF Export**: Optional PDF generation with customization hooks
- **LRU Caching**: Prompts are cached in memory for efficiency

## Docstring Coverage

| Item | Coverage |
|------|----------|
| Task.__init__ | Complete |
| build_message | Complete |
| validate_result | Complete with examples |
| add_dependency | Complete |
| run | Missing (needs improvement) |
| is_complete | Complete |
| get_prompt | Complete |
| get_result | Complete |
| save_as_pdf | Complete |
| post_process_result_before_saving_pdf | Complete |

Overall coverage: 82%. The critical `run()` method needs documentation.
