from enum import Enum
from typing import List, Optional, Set, Type, Union


class ModelFeatures(Enum):
    """
    Enum class to define the features supported by the models.
    """

    TextGeneration = "TextGeneration"  # Text generation feature
    StructuredTextGeneration = (
        "StructuredTextGeneration"  # Structured text generation feature
    )
    ImageGeneration = "ImageGeneration"  # Image generation feature
    AudioGeneration = "AudioGeneration"  # Audio generation feature
    VideoGeneration = "VideoGeneration"  # Video generation feature
    Thinking = "Thinking"  # Thinking feature, used for models that can think
    TextEmbeddings = "TextEmbeddings"  # Text embeddings feature
    ImageEmbeddings = "ImageEmbeddings"  # Image embeddings feature
    AudioEmbeddings = "AudioEmbeddings"  # Audio embeddings feature
    VideoEmbeddings = "VideoEmbeddings"  # Video embeddings feature
    DocumentEmbeddings = "DocumentEmbeddings"  # Document embeddings feature
    Image = (
        "Image"  # Model supports images _natively_ (not just as a conversion feature)
    )
    Audio = (
        "Audio"  # Model supports audio _natively_ (not just as a conversion feature)
    )
    Video = (
        "Video"  # Model supports video _natively_ (not just as a conversion feature)
    )
    Document = "Documents"  # Model supports (some) documents _natively_ (not just as a conversion feature)
    Tools = "Tools"  # Model supports tools
    WebSearchTool = "WebSearchTool"  # Model supports built-in web search tool
    MCPTool = "MCPTool"  # Model supports built-in MCP tool
    AudioTranscription = "AudioTranscription"  # Model supports audio transcription
    ImageConversion = "ImageConversion"  # Model supports image media conversion to simpler media, e.g to text
    AudioConversion = "AudioConversion"  # Model supports audio conversion to simpler media, e.g to text
    VideoConversion = "VideoConversion"  # Model supports video conversion to simpler media, e.g to text, audio and/or images
    DocumentConversion = "DocumentConversion"  # Model supports document conversion to simpler media, e.g to text and/or images

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

        # Convert the list of strings to lower case:
        if all(isinstance(feature, str) for feature in features):
            features = [feature.lower() for feature in features]
        else:
            # If anything is not a string, ensure it is a string:
            features = [str(feature) for feature in features]

        # Create an empty list to hold the normalised enum features:
        normalised_features = []

        # Iterate over the provided features normalised to lower case strings:
        for feature in features:

            # By default we have not found the corresponding feature:
            found = False

            # Iterate over the ModelFeatures enums:
            for feature_enum in ModelFeatures:
                # Get the lower case string for the feature:
                feature_name = feature_enum.name.lower()

                # If the feature is found, add it to the normalised list:
                if feature_name == feature:
                    normalised_features.append(feature_enum)
                    found = True
                    break

            if not found:
                # If the feature is not found, raise an error:
                raise ValueError(
                    f"Unknown feature: {feature} should be one of {', '.join([f.name for f in ModelFeatures])}"
                )

        # Return the normalised list of features
        return normalised_features

    @staticmethod
    def get_supported_media_types(
        features: Union[str, List[str], "ModelFeatures", List["ModelFeatures"]],
    ) -> Set[Type["MediaBase"]]:
        """
        Get the list of MediaBase-derived classes that correspond to the media types
        a model with the given features can ingest.

        Parameters
        ----------
        features: Union[str, List[str], ModelFeatures, List[ModelFeatures]]
            The model features to analyze.

        Returns
        -------
        List[Type[MediaBase]]
            List of MediaBase-derived classes representing media types the model can process.
        """

        # Normalize the input features to a list of ModelFeatures enums
        features = ModelFeatures.normalise(features)

        # Map ModelFeatures to actual MediaBase-derived classes
        from litemind.media.media_base import MediaBase
        from litemind.media.types.media_audio import Audio
        from litemind.media.types.media_document import Document
        from litemind.media.types.media_image import Image
        from litemind.media.types.media_text import Text
        from litemind.media.types.media_video import Video

        media_type_set: Set[Type[MediaBase]] = set()

        if ModelFeatures.TextGeneration in features:
            media_type_set.add(Text)

        if ModelFeatures.Image in features:
            media_type_set.add(Image)

        if ModelFeatures.Audio in features:
            media_type_set.add(Audio)

        if ModelFeatures.Video in features:
            media_type_set.add(Video)

        if ModelFeatures.Document in features:
            media_type_set.add(Document)

        return media_type_set

    @staticmethod
    def get_features_needed_for_media_types(
        media_types: Set[Type["MediaBase"]],
    ) -> Set["ModelFeatures"]:
        """
        Get the set of ModelFeatures needed to process the given media types.
        Importanyt note: This does not return Conversion features, only features that reflect native model capabilities.

        Parameters
        ----------
        media_types: List[Type[MediaBase]]
            The media types to analyze.

        Returns
        -------
        Set[ModelFeatures]
            Set of ModelFeatures needed for the given media types.
        """
        features_needed = set()

        for media_type in media_types:
            from litemind.media.types.media_audio import Audio
            from litemind.media.types.media_document import Document
            from litemind.media.types.media_image import Image
            from litemind.media.types.media_text import Text
            from litemind.media.types.media_video import Video

            if issubclass(media_type, Text):
                features_needed.add(ModelFeatures.TextGeneration)
            elif issubclass(media_type, Image):
                features_needed.add(ModelFeatures.Image)
            elif issubclass(media_type, Audio):
                features_needed.add(ModelFeatures.Audio)
            elif issubclass(media_type, Video):
                features_needed.add(ModelFeatures.Video)
            elif issubclass(media_type, Document):
                features_needed.add(ModelFeatures.Document)

        return features_needed

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
