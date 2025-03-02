from enum import Enum


class BlockType(Enum):
    Text = "text"
    Json = "json"
    Code = "code"
    Object = "object"
    Image = "image"
    Audio = "audio"
    Video = "video"
    Document = "document"
    Table = "table"
    Tool = "tool"

    @staticmethod
    def from_str(label: str):
        # Convert to lower case and remove spaces:
        label = label.lower().replace(" ", "")

        # Check if the label is in the enum:
        if label in BlockType._value2member_map_:
            return BlockType._value2member_map_[label]

        # Raise an error if the label is not in the enum:
        raise ValueError(f"Unknown block type: {label}")

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
