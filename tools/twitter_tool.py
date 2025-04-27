import os

from autogen_core.tools import FunctionTool
from dotenv import load_dotenv
from tweety import Twitter

from models.post import Post

load_dotenv()

twitter_session = os.getenv('TWITTER_SESSION')
if twitter_session and twitter_session.strip():
    with open('session.tw_session', 'w') as f:
        f.write(twitter_session)

app = Twitter('session')

last_post_id = ''


async def fetch(user_id: str) -> list[Post]:
    global last_post_id
    try:
        posts = await app.get_tweets(user_id, cursor=last_post_id)
    except Exception as e:
        print(e)
        return []

    none_empty_posts = []

    for post in posts:
        if 'tweets' in post:
            latest_id = None
            latest_created_on = None
            combined_text = ""
            latest_url = ""
            poster = None

            for tweet in post.tweets:
                if tweet.text:
                    combined_text += tweet.text + "\n"
                if latest_created_on is None or tweet.created_on > latest_created_on:
                    latest_created_on = tweet.created_on
                    latest_id = tweet.id
                    latest_url = tweet.url
                    poster = tweet.author

            if combined_text and latest_id and latest_created_on and poster:
                none_empty_posts.append(
                    Post(latest_id, latest_created_on, combined_text.strip(), latest_url, poster.name,
                         poster.profile_url))
        elif post.text:
            none_empty_posts.append(Post(post.id, post.created_on, post.text,
                                         post.url, post.author.name, post.author.profile_url))

    last_post_id = posts.cursor_top

    return none_empty_posts


async def get_user_post(user_id: str) -> list[Post]:
    await app.connect()
    if app.me is None:
        try:
            await app.sign_in(os.getenv('TWITTER_USERNAME'), os.getenv('TWITTER_PASSWORD'))
        except Exception as e:
            print(e)
            raise e

    return await fetch(user_id)


twitter_user_post_tool = FunctionTool(get_user_post, description="用于获取用户Twitter的最新推文的工具")
