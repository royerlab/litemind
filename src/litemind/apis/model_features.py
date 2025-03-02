from enum import Enum
from typing import List, Optional, Union


class ModelFeatures(Enum):
    """
    Enum class to define the features supported by the models.
    """

    TextGeneration = "TextGeneration"
    StructuredTextGeneration = "StructuredTextGeneration"
    ImageGeneration = "ImageGeneration"
    AudioGeneration = "AudioGeneration"
    VideoGeneration = "VideoGeneration"
    Reasoning = ("Reasoning",)
    TextEmbeddings = "TextEmbeddings"
    ImageEmbeddings = "ImageEmbeddings"
    AudioEmbeddings = "AudioEmbeddings"
    VideoEmbeddings = "VideoEmbeddings"
    Image = "Image"
    Audio = "Audio"
    Video = "Video"
    Document = "Documents"
    Tools = "Tools"
    AudioTranscription = "AudioTranscription"
    VideoConversion = "VideoConversion"
    DocumentConversion = "DocumentConversion"

    # Method that takes a single strings, a list of strings, a single ModelFeatures enum or a list of ModelFeatures and normalises to a list of enums of this class, finds the right enums independently of case:
    @staticmethod
    def normalise(
        features: Union[str, List[str], "ModelFeatures", List["ModelFeatures"]],
    ) -> Optional[List["ModelFeatures"]]:
        """
        Normalise the input features to a list of ModelFeatures enums.

        Parameters
        ----------
        features: Union[str, List[str], ModelFeatures, List[ModelFeatures]]
            The features to normalise.

        Returns
        -------
        List[ModelFeatures]
            The normalised list of ModelFeatures enums.

        """
        # If no feature set is defined then pass-through None:
        if features is None:
            return None

        # If the input is a single string, convert it to a list:
        if isinstance(features, str):
            features = [features]

        # If the input is a single enum, convert it to a list:
        if isinstance(features, ModelFeatures):
            features = [features]

        # If it is a list of enums just return:
        if all(isinstance(feature, ModelFeatures) for feature in features):
            return features

        # If it is a list of strings, convert to a list of enums , ignoring case:
        return [ModelFeatures(feature) for feature in features]

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
