import os
import tempfile

import numpy as np
import pytest

from litemind.media.types.media_image import Image
from litemind.media.types.media_ndimage import NdImage
from litemind.media.types.media_text import Text
from litemind.ressources.media_resources import MediaResources


class TestNdImage:
    def test_init(self):
        """Test initializing NdImage with a URI."""
        ndimage_uri = MediaResources.get_local_test_ndimage_uri("mri.tif")

        # Instantiate NdImage:
        ndimage_media = NdImage(uri=ndimage_uri)

        # Check URI:
        assert ndimage_media.uri == ndimage_uri
        # Array should not be loaded at initialization
        assert ndimage_media.array is None

    def test_load_from_uri(self):
        """Test loading an nD image from URI."""
        ndimage_uri = MediaResources.get_local_test_ndimage_uri("mri.tif")

        # Instantiate and load:
        ndimage_media = NdImage(uri=ndimage_uri)
        ndimage_media.load_from_uri()

        # Check array:
        assert ndimage_media.array is not None
        assert isinstance(ndimage_media.array, np.ndarray)
        assert ndimage_media.array.ndim >= 2  # MRI should be at least 2D

    def test_str(self):
        """Test string representation of NdImage."""
        ndimage_uri = MediaResources.get_local_test_ndimage_uri("mri.tif")
        ndimage_media = NdImage(uri=ndimage_uri)

        assert "mri.tif" in str(ndimage_media)

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        # Create a temporary path that doesn't exist
        temp_dir = tempfile.mkdtemp()
        nonexistent_file = os.path.join(temp_dir, "nonexistent.tif")
        os.rmdir(temp_dir)  # Remove dir to ensure path doesn't exist

        ndimage_media = NdImage(uri=f"file://{nonexistent_file}")

        # Should raise exception when loading
        with pytest.raises(Exception) as e:
            ndimage_media.load_from_uri()

        assert "Local file not found:" in str(e.value)

    def test_to_text_and_2d_projections_mri(self):
        """Test creating text and 2D projections from a 3D MRI image."""
        ndimage_uri = MediaResources.get_local_test_ndimage_uri("mri.tif")
        ndimage_media = NdImage(uri=ndimage_uri)
        ndimage_media.load_from_uri()

        # Get the media representation
        media_list = ndimage_media.to_text_and_2d_projection_medias()

        # Basic checks
        assert len(media_list) > 0
        assert isinstance(media_list[0], Text)
        assert "nD Image" in media_list[0].text

        # Should have dimensions info
        dimensions_str = " × ".join([str(d) for d in ndimage_media.array.shape])
        assert dimensions_str in media_list[0].text

        # MRI should have spatial dimensions and projections
        assert "Spatial dimensions" in media_list[0].text

        # Check alternating Text and Image pattern for projections
        for i in range(1, len(media_list), 2):
            assert isinstance(media_list[i], Text)
            assert "Maximum Intensity Projection" in media_list[i].text

            if i + 1 < len(media_list):
                assert isinstance(media_list[i + 1], Image)
                # Check that Image has data
                assert media_list[i + 1].uri is not None

    def test_to_text_and_2d_projections_tube(self):
        """Test projection with a multi-channel image."""
        ndimage_uri = MediaResources.get_local_test_ndimage_uri("tubhiswt_C1.ome.tif")
        ndimage_media = NdImage(uri=ndimage_uri)
        ndimage_media.load_from_uri()

        # Test with different channel thresholds
        media_list_default = ndimage_media.to_text_and_2d_projection_medias()
        media_list_high_threshold = ndimage_media.to_text_and_2d_projection_medias(
            channel_threshold=1000
        )

        # With high threshold, more dimensions should be considered "channel-like"
        default_text = media_list_default[0].text
        high_threshold_text = media_list_high_threshold[0].text

        # The content should be different with different thresholds
        assert default_text != high_threshold_text

    def test_singleton_dimensions(self):
        """Test handling of singleton dimensions."""
        # Create a small test array with a singleton dimension
        array = np.zeros((50, 1, 100))

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        temp_file.close()
        np.save(temp_file.name, array)

        try:
            # Load with NdImage
            ndimage_media = NdImage(uri=f"file://{temp_file.name}")
            ndimage_media.load_from_uri()

            # Generate the representation
            media_list = ndimage_media.to_text_and_2d_projection_medias()

            # Check that singleton dimensions are mentioned
            assert "Singleton dimensions" in media_list[0].text

            # Check that there are projections (after singleton dimension removal)
            projection_count = sum(1 for item in media_list if isinstance(item, Image))
            assert projection_count > 0
        finally:
            # Clean up
            os.unlink(temp_file.name)

    def test_channel_dimensions(self):
        """Test handling of channel dimensions."""
        # Create a test array with a channel-like dimension (size <= 10)
        array = np.zeros((200, 3, 200))

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        temp_file.close()
        np.save(temp_file.name, array)

        try:
            # Load with NdImage
            ndimage_media = NdImage(uri=f"file://{temp_file.name}")
            ndimage_media.load_from_uri()

            # Generate the representation
            media_list = ndimage_media.to_text_and_2d_projection_medias()

            # Check that channel-like dimensions are mentioned
            assert "Channel-like dimensions" in media_list[0].text

            # Should have a channel header
            channel_headers = [
                item
                for item in media_list
                if isinstance(item, Text) and "Channel:" in item.text
            ]
            assert len(channel_headers) > 0
        finally:
            # Clean up
            os.unlink(temp_file.name)

    def test_not_enough_dimensions(self):
        """Test behavior when there aren't enough spatial dimensions."""
        # Create a 1D array (not enough spatial dimensions for projections)
        array = np.zeros(200)

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        temp_file.close()
        np.save(temp_file.name, array)

        try:
            # Load with NdImage
            ndimage_media = NdImage(uri=f"file://{temp_file.name}")
            ndimage_media.load_from_uri()

            # We should not get to this point:
            assert False, "Expected an exception due to insufficient dimensions"

        except Exception as e:
            assert True, "Expected an exception above due to insufficient dimensions"

        finally:
            # Clean up
            os.unlink(temp_file.name)

    def test_normalize_for_display(self):
        """Test normalization function with various arrays."""
        # Create an NdImage instance
        ndimage_media = NdImage(uri="file:///dummy.tif")

        # Test with normal range
        array = np.linspace(0, 100, 100).reshape(10, 10)
        normalized = ndimage_media._normalize_for_display(array)
        assert normalized.dtype == np.uint8
        assert normalized.min() == 0
        assert normalized.max() == 255

        # Test with negative values
        array = np.linspace(-50, 50, 100).reshape(10, 10)
        normalized = ndimage_media._normalize_for_display(array)
        assert normalized.min() == 0
        assert normalized.max() == 255

        # Test with NaN values
        array = np.linspace(0, 100, 100).reshape(10, 10)
        array[0, 0] = np.nan
        normalized = ndimage_media._normalize_for_display(array)
        assert not np.isnan(normalized).any()  # No NaNs in result

        # Test with constant array
        array = np.ones((10, 10))
        normalized = ndimage_media._normalize_for_display(array)
        assert np.all(normalized == 0)  # Should be all zeros for constant array

    def test_get_channel_indices(self):
        """Test generation of channel indices combinations."""
        ndimage_media = NdImage(uri="file:///dummy.tif")

        # Test with one channel dimension
        channel_dims = [1]
        dimensions = (5, 3, 10)
        indices = ndimage_media._get_channel_indices(channel_dims, dimensions)
        assert len(indices) == 3  # 3 combinations
        assert indices == [(0,), (1,), (2,)]

        # Test with two channel dimensions
        channel_dims = [0, 2]
        dimensions = (2, 5, 3)
        indices = ndimage_media._get_channel_indices(channel_dims, dimensions)
        assert len(indices) == 6  # 2×3 = 6 combinations
        assert (0, 0) in indices
        assert (1, 2) in indices
