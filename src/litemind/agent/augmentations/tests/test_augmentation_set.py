from litemind.agent.augmentations.augmentation_base import AugmentationBase
from litemind.agent.augmentations.augmentation_set import AugmentationSet
from litemind.agent.augmentations.information.information import Information


class DummyAugmentation(AugmentationBase):
    def __init__(self, name):
        self.name = name

    def get_relevant_informations(self, query, k=5, threshold=0.0):
        return [
            Information(media=f"Result {i} for {query}", metadata={"score": i})
            for i in range(k)
        ]


def test_add_augmentation():
    aug_set = AugmentationSet()
    dummy_aug = DummyAugmentation(name="dummy")
    aug_set.add_augmentation(dummy_aug)
    assert len(aug_set) == 1
    assert aug_set.get_augmentation("dummy") == dummy_aug


def test_get_augmentation():
    dummy_aug = DummyAugmentation(name="dummy")
    aug_set = AugmentationSet()
    aug_set.add_augmentation(dummy_aug)
    assert aug_set.get_augmentation("dummy") == dummy_aug
    assert aug_set.get_augmentation("nonexistent") is None


def test_list_augmentations():
    dummy_aug1 = DummyAugmentation(name="dummy1")
    dummy_aug2 = DummyAugmentation(name="dummy2")
    aug_set = AugmentationSet()
    aug_set.add_augmentation(dummy_aug1)
    aug_set.add_augmentation(dummy_aug2)
    assert aug_set.list_augmentations() == [dummy_aug1, dummy_aug2]


def test_augmentation_names():
    dummy_aug1 = DummyAugmentation(name="dummy1")
    dummy_aug2 = DummyAugmentation(name="dummy2")
    aug_set = AugmentationSet()
    aug_set.add_augmentation(dummy_aug1)
    aug_set.add_augmentation(dummy_aug2)
    assert aug_set.augmentation_names() == ["dummy1", "dummy2"]


def test_remove_augmentation():
    dummy_aug = DummyAugmentation(name="dummy")
    aug_set = AugmentationSet()
    aug_set.add_augmentation(dummy_aug)
    assert aug_set.remove_augmentation("dummy") is True
    assert len(aug_set) == 0
    assert aug_set.remove_augmentation("nonexistent") is False


def test_search_all():
    dummy_aug1 = DummyAugmentation(name="dummy1")
    dummy_aug2 = DummyAugmentation(name="dummy2")
    aug_set = AugmentationSet()
    aug_set.add_augmentation(dummy_aug1)
    aug_set.add_augmentation(dummy_aug2)
    results = aug_set.search_all("test query", k=3)
    assert "dummy1" in results
    assert "dummy2" in results
    assert len(results["dummy1"]) == 3
    assert len(results["dummy2"]) == 3


def test_search_combined():
    dummy_aug1 = DummyAugmentation(name="dummy1")
    dummy_aug2 = DummyAugmentation(name="dummy2")
    aug_set = AugmentationSet()
    aug_set.add_augmentation(dummy_aug1)
    aug_set.add_augmentation(dummy_aug2)
    results = aug_set.search_combined("test query", k=5)
    assert len(results) == 5
    assert all(doc.metadata["augmentation"] in ["dummy1", "dummy2"] for doc in results)


def test_len():
    dummy_aug = DummyAugmentation(name="dummy")
    aug_set = AugmentationSet()
    aug_set.add_augmentation(dummy_aug)
    assert len(aug_set) == 1


def test_iter():
    dummy_aug1 = DummyAugmentation(name="dummy1")
    dummy_aug2 = DummyAugmentation(name="dummy2")
    aug_set = AugmentationSet()
    aug_set.add_augmentation(dummy_aug1)
    aug_set.add_augmentation(dummy_aug2)
    aug_list = [aug for aug in aug_set]
    assert aug_list == [dummy_aug1, dummy_aug2]


def test_contains():
    dummy_aug = DummyAugmentation(name="dummy")
    aug_set = AugmentationSet()
    aug_set.add_augmentation(dummy_aug)
    assert "dummy" in aug_set
    assert dummy_aug in aug_set
    assert "nonexistent" not in aug_set


def test_str_repr():
    dummy_aug1 = DummyAugmentation(name="dummy1")
    dummy_aug2 = DummyAugmentation(name="dummy2")
    aug_set = AugmentationSet()
    aug_set.add_augmentation(dummy_aug1)
    aug_set.add_augmentation(dummy_aug2)
    assert str(aug_set) == "AugmentationSet(['dummy1', 'dummy2'])"
    assert repr(aug_set) == "AugmentationSet(['dummy1', 'dummy2'])"
