from abc import ABC, abstractmethod
from typing import List, Optional

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet


class BaseApi(ABC):

    @abstractmethod
    def check_api_key(self, api_key: Optional[str]) -> bool:
        pass

    @abstractmethod
    def model_list(self) -> List[str]:
        pass

    @abstractmethod
    def default_model(self, require_vision: bool = False) -> str:
        pass

    @abstractmethod
    def has_vision_support(self, model_name: Optional[str] = None) -> bool:
        pass

    @abstractmethod
    def has_tool_support(self, model_name: Optional[str] = None) -> bool:
        pass

    @abstractmethod
    def max_num_input_token(self, model_name: Optional[str] = None) -> int:
        pass

    @abstractmethod
    def completion(self,
                   model_name: str,
                   messages: List[Message],
                   temperature: float,
                   toolset: Optional[ToolSet] = None,
                   **kwargs) -> Message:
        pass

    @abstractmethod
    def describe_image(self,
                       image_path: str,
                       query: str = 'Here is an image, please carefully describe it in detail.',
                       model_name: str = "gpt-4-vision-preview",
                       max_tokens: int = 4096,
                       number_of_tries: int = 4,
                       ) -> str:
        pass
