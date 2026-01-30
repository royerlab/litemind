# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install for development (with all optional dependencies)
pip install -e ".[dev,rag,whisper,documents,tables,videos,audio,remote,tasks]"

# Format code
isort .
black .

# Lint and type check
flake8 .
mypy src/litemind

# Run all tests
pytest src/

# Run a single test file
pytest src/litemind/apis/tests/test_openai_api.py

# Run tests matching a pattern
pytest -k "tools" src/

# Run tests with coverage report
pytest --cov=src --cov-report=html:reports/coverage src/

# Build package
hatch clean && hatch build
```

## CLI Tools

```bash
# Export repository to single file
litemind export --folder-path . --output-file exported.txt

# Validate model registry against live APIs
litemind validate --api openai gemini --models gpt-4o models/gemini-1.5-pro

# Discover features for unknown models
litemind discover --api openai --models new-model-name
```

## Architecture Overview

Litemind is a unified API wrapper and agentic AI framework supporting multiple LLM providers (OpenAI, Anthropic, Google Gemini, Ollama).

### Core Layers

**API Layer** (`src/litemind/apis/`):
- `BaseApi` → `DefaultApi` → Provider implementations (OpenAIApi, AnthropicApi, GeminiApi, OllamaApi)
- `CombinedApi`: Chains multiple APIs for fallback
- Feature-based model selection via `ModelFeatures` enum and `get_best_model(features=[...])`
- Provider-specific code lives in `providers/<provider>/` with standard methods: `convert_messages`, `format_tools`, `process_response`

**Agent Layer** (`src/litemind/agent/`):
- `Agent`: Main orchestrator managing conversation state, tool execution, and augmentations
- `Conversation`: Manages system and user messages
- `Message`/`MessageBlock`: Multimodal message containers
- `ToolSet`: Collection of tools (FunctionTool, AgentTool, BuiltinTool)
- `AugmentationSet`: RAG augmentations including vector databases

**Media Layer** (`src/litemind/media/`):
- `MediaBase` subclasses: `Text`, `Code`, `Json`, `Image`, `Audio`, `Video`, `Table`, `Document`, `NdImage`
- `MediaConverter`: Automatic format conversion pipeline for model compatibility
- Conversion utilities in `media/conversion/`

**Augmentations** (`src/litemind/agent/augmentations/`):
- `BaseVectorDB` → `InMemoryVectorDatabase`, `QdrantVectorDatabase`
- `Information`: Knowledge units with metadata for RAG

### Key Patterns

- Abstract base classes define interfaces (`BaseApi`, `BaseTool`, `AugmentationBase`, `MediaBase`)
- Callback system throughout (`ApiCallbackManager`, `ToolCallbackManager`)
- Feature matrices describe model capabilities for intelligent selection
- Tests mirror source structure (e.g., `apis/tests/`, `agent/tests/`, `media/tests/`)

### Model Registry Architecture

Model feature data is curated in YAML files (`src/litemind/apis/model_registry/*.yaml`) rather than auto-discovered via live API calls. This design decision was made because:
- Auto-discovery (the old scanner approach) produced false negatives and brittle tests
- Curated YAML provides accurate feature data, token limits, aliases, and metadata
- Each provider has a registry file: `registry_openai.yaml`, `registry_anthropic.yaml`, `registry_gemini.yaml`, `registry_ollama.yaml`
- Use `ModelRegistry.supports_feature()`, `get_model_info()`, and `resolve_alias()` for feature lookups
- The `litemind validate` CLI can validate registry accuracy against live APIs when needed

### Data Flow

```
User Message → Agent → Augmentation retrieval → Message formatting →
API call → Tool execution (if needed) → Response → Conversation history
```

## Environment Variables

- `OPENAI_API_KEY` - OpenAI API
- `ANTHROPIC_API_KEY` - Anthropic/Claude API
- `GOOGLE_GEMINI_API_KEY` - Google Gemini API
- Ollama: No key required (local server)

## Code Style

- Use Black for formatting, isort for imports
- Numpy-style docstrings
- Type hints throughout
- Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`

## Documentation Conventions

### Sub-package README Files

Each major sub-package should have a `README.md` file explaining its purpose, structure, and usage. Current sub-packages with READMEs:

- `src/litemind/agent/README.md` - Agentic AI framework (Agent, Messages, Tools, Augmentations)
- `src/litemind/apis/README.md` - API abstraction layer (providers, model registry, features)
- `src/litemind/media/README.md` - Multimodal media types and conversion system
- `src/litemind/utils/README.md` - Utility functions (file handling, embeddings, etc.)
- `src/litemind/tools/README.md` - CLI commands (export, validate)
- `src/litemind/workflow/README.md` - Task workflow system
- `src/litemind/remote/README.md` - Remote agent execution via RPyC

### README Content Guidelines

Each sub-package README should include:
1. **Package Structure** - Directory tree showing key files
2. **Core Components** - Classes and functions with brief descriptions
3. **Usage Examples** - Code snippets demonstrating common patterns
4. **Docstring Coverage** - Summary table of documentation completeness
5. **Dependencies** - External libraries required (if applicable)

## Main README Generation

When updating the main `README.md`, use Claude Code directly. The README should follow this structure:

### README Sections

1. **Summary** - Enthusiastic and complete description of the library, its purpose, and philosophy
2. **Features** - List of main features including unique or standout capabilities
3. **Installation** - How to install litemind
4. **Basic Usage** - Several illustrative examples of the agent-level API (with multimodal inputs):
   - Basic agent usage
   - Agent with tools
   - Agent with tools and augmentation (RAG)
   - Complex example with multimodal inputs, tools and augmentations
5. **Concepts** - Main classes, their purpose, and interactions. Explain the difference between the API wrapper layer versus the agentic API
6. **Multi-modality** - Multimodal capabilities, ModelFeatures, how to request features, Media classes
7. **More Examples** - Advanced use cases:
   - CombinedAPI usage
   - Agent that can execute Python code
   - Wrapper API for structured outputs
   - Agent with tool that generates images
8. **Command Line Tools** - CLI tools with example command lines
9. **Caveats and Limitations** - Known issues or areas for improvement
10. **Code Health** - Unit test results from `test_report.md` (total tests, passed, failed, assessment)
11. **API Keys** - How to obtain and configure API keys for each provider (OpenAI, Claude, Gemini). Which environment variables to set for each OS
12. **Roadmap** - Use contents of TODO.md as starting point
13. **Contributing** - Refer to CONTRIBUTING.md
14. **License**

### Code Examples Guidelines

When writing code examples in the README:
- Use markdown code blocks
- Keep different examples in separate code blocks with text preambles
- Examples must be complete and runnable as-is
- Include all required imports
- Include expected output as comments where possible
- Make examples didactic, well-commented, and easy to understand
- Follow PEP-8 formatting
- Code examples using specific features must request models that have those features
- Use real URLs of files that exist (e.g., from the tests)
- Show that ModelFeatures can be provided as strings instead of enums (see normalisation code)

### Important Notes

- Add badges for: PyPI, license, pepy.tech stats, GitHub stars
- The Pepy.tech badge image: `https://static.pepy.tech/badge/litemind`
- Explain that logging uses Arbol (http://github.com/royerlab/arbol) and can be deactivated with `Arbol.passthrough = True`
- **Avoid hallucinating** - Only document features that exist in the repository
- Litemind does **NOT** have a `ReActAgent` class or anything 'react' related - the `Agent` class itself has all ReAct framework features
- If unsure about something, don't make it up
