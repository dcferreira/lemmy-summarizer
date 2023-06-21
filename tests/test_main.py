import pytest
from polyfactory.factories.pydantic_factory import ModelFactory
from pylemmy import Lemmy
from pylemmy.api.post import PostView
from pylemmy.models.post import Post

from lemmy_summarizer.main import summarize_post


@pytest.fixture
def lemmy():
    return Lemmy(
        lemmy_url="http://127.0.0.1:8536",
        username=None,
        password=None,
        user_agent="summarizer (by github.com/dcferreira)",
    )


@pytest.fixture
def fake_summarizer():
    def summarizer(text, *args, **kwargs):  # noqa: ARG001
        # does nothing
        return [{"summary_text": text}]

    return summarizer


class PostViewFactory(ModelFactory[PostView]):
    __model__ = PostView


def test_summarize_post_fail_request(lemmy, fake_summarizer):
    url = "https://example.com/bad_url"
    post_view = PostViewFactory.build()
    post_view.post.url = url
    post = Post(lemmy, post_view)
    out = summarize_post(post, fake_summarizer, 0)
    assert out is None


def test_summarize_post_no_words(lemmy, fake_summarizer):
    url = "https://www.google.com/favicon.ico"
    post_view = PostViewFactory.build()
    post_view.post.url = url
    post = Post(lemmy, post_view)
    out = summarize_post(post, fake_summarizer, 0)
    assert out is None


def test_summarize_post_few_words(lemmy, fake_summarizer):
    url = "https://www.google.com"
    post_view = PostViewFactory.build()
    post_view.post.url = url
    post = Post(lemmy, post_view)
    out = summarize_post(post, fake_summarizer, 10000)
    assert out is None


def test_summarize_post_summarize(lemmy, fake_summarizer):
    url = "https://www.bbc.com"
    post_view = PostViewFactory.build()
    post_view.post.url = url
    post = Post(lemmy, post_view)
    out = summarize_post(post, fake_summarizer, 100)
    assert isinstance(out, str)
