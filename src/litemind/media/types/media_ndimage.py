from typing import List, Optional, Union

import numpy
from arbol import aprint

from litemind.media.media_uri import MediaURI
from litemind.media.types.media_image import Image
from litemind.media.types.media_text import Text
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


class NdImage(MediaURI):
    """
    A media that stores an nD image, aka a multi-dimensional image.
    """

    def __init__(self, uri: str, extension: Optional[str] = None, **kwargs):
        """
        Create a new nD image media.

        Parameters
        ----------
        uri: str
            The image URI.
        extension: str
            Extension/Type of the nD image file in case it is not clear from the URI. This is the extension _without_ the dot -- 'png' not '.png'.
        kwargs: dict
            Other arguments passed to MediaURI.

        """

        super().__init__(uri=uri, extension=extension, **kwargs)

        # Set attributes:
        self.array = None

    def load_from_uri(self):
        # Download the file from the URI to a local file:
        local_file = uri_to_local_file_path(self.uri)

        try:
            # Handle numpy files directly
            if local_file.lower().endswith(".npy"):
                import numpy as np

                self.array = np.load(local_file)
            elif local_file.lower().endswith(".npz"):
                import numpy as np

                # For .npz files, load the first array in the archive
                with np.load(local_file) as data:
                    array_names = list(data.keys())
                    if not array_names:
                        raise Exception("NPZ file contains no arrays")
                    self.array = data[array_names[0]]
            # Handle tiff files with tifffile
            elif local_file.lower().endswith((".tif", ".tiff")):
                import tifffile

                self.array = tifffile.imread(local_file)
            # For other formats use imageio
            else:
                import imageio.v3 as iio

                self.array = iio.imread(local_file)

            # Print debug info about loaded array
            aprint(
                f"Loaded array with shape: {self.array.shape}, dtype: {self.array.dtype}"
            )

            # Ensure we have at least 2D array
            if self.array.ndim < 2:
                raise Exception(
                    f"Image has only {self.array.ndim} dimensions, expected at least 2."
                )

        except Exception as e:
            raise Exception(
                f"Could not open or decode nD image file: {self.uri}. Error: {e}"
            )

    def to_text_and_2d_projection_medias(
        self, channel_threshold=10
    ) -> List[Union[Text, Image]]:
        """
        Creates a list of Text and Image media objects that describe the nD image
        for LLM ingestion purposes. Handles singleton dimensions and channel-like dimensions specially.

        Parameters
        ----------
        channel_threshold : int
            Dimensions with size less than or equal to this threshold are considered channel-like

        Returns
        -------
        list
            A list of alternating Text and Image media objects:
            - First a Text with markdown description of the array
            - Then pairs of Text and Image objects for each maximum projection
        """

        import numpy as np

        result = []

        # Ensure the array is loaded
        if self.array is None:
            self.load_from_uri()

        # Part 1: Create markdown description of the array
        dimensions = self.array.shape
        dimensions_str = " × ".join([str(d) for d in dimensions])
        ndim = self.array.ndim
        dtype = str(self.array.dtype)
        min_val = np.nanmin(self.array)
        max_val = np.nanmax(self.array)

        # Filename:
        filename = self.get_filename()

        description = f"""## nD Image '{filename}'

    **Dimensions:** {dimensions_str}
    **Number of dimensions:** {ndim}
    **Data type:** {dtype}
    **Value range:** [{min_val:.3g}, {max_val:.3g}]
    """

        # Add statistics if not too large
        if self.array.size < 1_000_000:  # Only for reasonably sized arrays
            mean_val = np.nanmean(self.array)
            median_val = np.nanmedian(self.array)
            std_val = np.nanstd(self.array)
            description += f"""**Mean:** {mean_val:.3g}
    **Median:** {median_val:.3g}
    **Standard deviation:** {std_val:.3g}
    """

        # Analyze dimensions
        singleton_dims = [i for i, size in enumerate(dimensions) if size == 1]
        channel_like_dims = [
            i for i, size in enumerate(dimensions) if 1 < size <= channel_threshold
        ]
        spatial_dims = [
            i
            for i, size in enumerate(dimensions)
            if size > channel_threshold and i not in singleton_dims
        ]

        # Add dimension analysis
        if singleton_dims:
            description += f"\n**Singleton dimensions (ignored):** {', '.join([str(d) for d in singleton_dims])}\n"
        if channel_like_dims:
            description += f"\n**Channel-like dimensions:** {', '.join([str(d) for d in channel_like_dims])}\n"
        if spatial_dims:
            description += f"\n**Spatial dimensions:** {', '.join([str(d) for d in spatial_dims])}\n"

        result.append(Text(text=description))

        # If we have less than 2 spatial dimensions, we can't create 2D projections
        if len(spatial_dims) < 2:
            result.append(
                Text(
                    text="**Note:** Not enough spatial dimensions to create 2D projections."
                )
            )
            return result

        # Create a common set of dimension labels
        dim_labels = ["x", "y", "z", "t", "c", "s"]
        # Fill with generic names if needed
        while len(dim_labels) < ndim:
            dim_labels.append(f"dim{len(dim_labels)}")

        # Get all pairs of spatial dimensions for projections
        spatial_pairs = []
        for i, dim1 in enumerate(spatial_dims):
            for j in range(i + 1, len(spatial_dims)):
                dim2 = spatial_dims[j]
                spatial_pairs.append((dim1, dim2))

        # If we have channel-like dimensions, iterate through them
        if channel_like_dims:
            # Create all combinations of channel indices
            channel_indices = self._get_channel_indices(channel_like_dims, dimensions)

            for channel_idx in channel_indices:
                # Create channel description
                channel_desc = []
                for dim, idx in zip(channel_like_dims, channel_idx):
                    channel_desc.append(f"{dim_labels[dim]}={idx}")

                channel_text = f"### Channel: {', '.join(channel_desc)}\n"
                result.append(Text(text=channel_text))

                # Create projections for this channel
                for axis1, axis2 in spatial_pairs:
                    # Create the projection with specified channel indices
                    self._add_projection_to_result(
                        axis1,
                        axis2,
                        dim_labels,
                        result,
                        channel_dims=channel_like_dims,
                        channel_indices=channel_idx,
                    )
        else:
            # No channel dimensions, just do projections for spatial dimensions
            for axis1, axis2 in spatial_pairs:
                self._add_projection_to_result(axis1, axis2, dim_labels, result)

        return result

    def _get_channel_indices(self, channel_dims, dimensions):
        """
        Get all possible combinations of indices for channel dimensions.

        Parameters
        ----------
        channel_dims : list
            List of dimension indices that are considered channels
        dimensions : tuple
            Shape of the array

        Returns
        -------
        list
            List of tuples with channel indices combinations
        """
        import itertools

        # Get all possible index values for each channel dimension
        channel_values = [range(dimensions[dim]) for dim in channel_dims]

        # Create all combinations of indices
        return list(itertools.product(*channel_values))

    def _add_projection_to_result(
        self, axis1, axis2, dim_labels, result, channel_dims=None, channel_indices=None
    ):
        """
        Creates and adds a projection for the specified axes to the result list.

        Parameters
        ----------
        axis1, axis2 : int
            Indices of axes to project onto
        dim_labels : list
            Labels for dimensions
        result : list
            List to append the Text and Image objects to
        channel_dims : list, optional
            Indices of channel-like dimensions
        channel_indices : tuple, optional
            Index values for each channel dimension
        """
        import tempfile

        from litemind.media.types.media_image import Image
        from litemind.media.types.media_text import Text

        axis1_label = dim_labels[axis1]
        axis2_label = dim_labels[axis2]

        # Create description text
        proj_description = f"""### Maximum Intensity Projection: {axis1_label}{axis2_label}-plane

    This image shows the maximum intensity projection along the {axis1_label}{axis2_label}-plane
    (dimensions {axis1} and {axis2} of the original array).
    """
        result.append(Text(proj_description))

        # Create the projection image with channel information if provided
        projection = self._create_max_projection(
            axis1, axis2, channel_dims=channel_dims, channel_indices=channel_indices
        )

        # Save as PNG and create Image media
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_file.close()

        # Convert projection to Image media object
        proj_image = Image.from_data(projection, filepath=temp_file.name, format="PNG")
        result.append(proj_image)

    def _create_max_projection(
        self, axis1, axis2, channel_dims=None, channel_indices=None
    ):
        """
        Creates a maximum intensity projection along two specified axes.
        """
        import numpy as np

        # Ensure the array is loaded
        if self.array is None:
            self.load_from_uri()

        # Create a view with the specified channel indices if channels are provided
        if channel_dims is not None and channel_indices is not None:
            slices = [slice(None)] * self.array.ndim
            for dim_idx, channel_idx in zip(channel_dims, channel_indices):
                slices[dim_idx] = channel_idx
            array_view = self.array[tuple(slices)]
        else:
            array_view = self.array

        # Identify dimensions of size 1 (singleton dimensions)
        singleton_dims = [i for i, size in enumerate(array_view.shape) if size == 1]

        # Squeeze out singleton dimensions
        if singleton_dims:
            array_view = np.squeeze(array_view, axis=tuple(singleton_dims))
            # Adjust axis1 and axis2 if they were affected by the squeeze
            for dim in sorted(singleton_dims):
                if axis1 > dim:
                    axis1 -= 1
                if axis2 > dim:
                    axis2 -= 1

        # Get actual dimensionality
        ndim = array_view.ndim

        # Check if axes are valid
        if axis1 >= ndim or axis2 >= ndim:
            # Invalid axes for this array shape
            error_img = np.zeros((100, 100), dtype=np.uint8)
            return np.stack([error_img] * 3, axis=-1)  # Return blank RGB image

        # Identify the axes to project along
        projection_axes = tuple([i for i in range(ndim) if i != axis1 and i != axis2])

        # Create maximum projection
        if projection_axes:
            projection = np.max(array_view, axis=projection_axes)
        else:
            projection = array_view  # Already 2D

        # After max projection, we should have a 2D result
        # Check the projection shape and adjust
        if projection.ndim > 2:
            # If we still have more than 2D, take the first slice of higher dimensions
            projection = projection[(0,) * (projection.ndim - 2)]

        # Normalize to [0, 255] for display
        projection = self._normalize_for_display(projection)

        # Convert to RGB
        if projection.ndim == 2:
            rgb_projection = np.stack([projection] * 3, axis=-1)
        elif projection.ndim == 3 and projection.shape[2] == 3:
            rgb_projection = projection
        else:
            # Handle any unexpected cases
            rgb_projection = np.stack([projection[:, :, 0]] * 3, axis=-1)

        return rgb_projection.astype(np.uint8)

    def _normalize_for_display(self, array):
        """
        Normalizes array values to [0, 255] range with improved contrast.

        Parameters
        ----------
        array : numpy.ndarray
            Input array to normalize

        Returns
        -------
        numpy.ndarray
            Normalized array in uint8 format
        """
        import numpy as np

        # Make a copy to avoid modifying the original
        normalized = array.copy()

        # Handle special cases
        if np.all(normalized == normalized.flat[0]):  # All values are the same
            return np.zeros_like(normalized, dtype=np.uint8)

        # Get min/max, excluding NaN/Inf
        valid_data = normalized[np.isfinite(normalized)]
        if len(valid_data) == 0:  # All NaN/Inf
            return np.zeros_like(normalized, dtype=np.uint8)

        vmin, vmax = np.nanmin(valid_data), np.nanmax(valid_data)

        # Optional: Improve contrast with percentile-based normalization
        p_low, p_high = 0.5, 99.5  # Percentiles for contrast stretching
        if normalized.size > 100:  # Only for reasonably sized arrays
            vmin, vmax = np.nanpercentile(valid_data, [p_low, p_high])

        # Replace NaN with min value
        normalized = np.nan_to_num(normalized, nan=vmin)

        # Clip to range and normalize
        normalized = np.clip(normalized, vmin, vmax)

        # Scale to [0, 255]
        if vmax > vmin:
            normalized = ((normalized - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(normalized, dtype=np.uint8)

        return normalized
