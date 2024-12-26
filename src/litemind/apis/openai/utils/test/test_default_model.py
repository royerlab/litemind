from litemind.apis.openai.utils.default_model import \
    get_default_openai_model_name


def test_get_default_openai_model_name():
    result = get_default_openai_model_name()
    assert result
    assert result.startswith('gpt-')
