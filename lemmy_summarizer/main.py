"""Summarizer bot using facebook/bart-large-cnn.

This is a simple bot that monitors posts in a specific community and tries
to summarize them.
"""
from typing import Optional

import trafilatura
from loguru import logger
from pylemmy import Lemmy
from pylemmy.api.listing import SortType
from pylemmy.models.post import Post
from trafilatura.downloads import RawResponse
from transformers import Pipeline, pipeline


def main():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    lemmy = Lemmy(
        # lemmy_url="http://127.0.0.1:8536",
        # username="lemmy",
        # password="lemmylemmy",
        lemmy_url="https://beehaw.org",
        username=None,
        password=None,
        user_agent="summarizer (by github.com/dcferreira)",
    )

    community = lemmy.get_community("News")
    for post in community.stream.get_posts(sort=SortType.New):
        process_post(post, summarizer)


def summarize_post(
    post: Post, summarizer: Pipeline, min_text_size: int
) -> Optional[str]:
    # skip posts without a url
    if post.post_view.post.url is None:
        return None

    title = post.post_view.post.name
    ap_id = post.post_view.post.ap_id
    post_info_format = f"'{ap_id}': {title}"
    url = post.post_view.post.url

    downloaded: Optional[RawResponse] = trafilatura.fetch_url(url)
    if downloaded is None:
        logger.info(f"Failed to download {post_info_format}")
        return None
    extracted_text: Optional[str] = trafilatura.extract(downloaded)
    if extracted_text is None:
        logger.info(f"Skipping due to no text found {post_info_format}")
        return None
    # skip summarizing when the text is too small
    if len(extracted_text) < min_text_size:
        logger.info(
            f"Skipping due to too short text ({len(extracted_text)} chars) "
            f"{post_info_format}"
        )
        return None

    logger.info(f"Summarizing: {post_info_format}")
    summary = summarizer(
        extracted_text,
        max_new_tokens=300,
        min_new_tokens=100,
        truncation=True,
        do_sample=False,
    )[0]
    summary_text = summary["summary_text"].strip()
    logger.debug(f"Summary for {post_info_format}:\n{summary_text}")

    return summary_text


def process_post(post: Post, summarizer: Pipeline, min_text_size: int = 500):
    summary_text = summarize_post(
        post, summarizer=summarizer, min_text_size=min_text_size
    )
    if summary_text is None:
        return

    comment_text = f"""
This is a summary of the posted article (I'm a bot).

> {summary_text}

[How do I work?](https://github.com/dcferreira/lemmy-summarizer)"""

    print(comment_text)  # noqa: T201
    # post.create_comment(comment_text)


if __name__ == "__main__":
    main()
