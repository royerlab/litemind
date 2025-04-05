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
    Thinking = ("Thinking",)
    TextEmbeddings = "TextEmbeddings"
    ImageEmbeddings = "ImageEmbeddings"
    AudioEmbeddings = "AudioEmbeddings"
    VideoEmbeddings = "VideoEmbeddings"
    DocumentEmbeddings = "DocumentEmbeddings"
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

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
