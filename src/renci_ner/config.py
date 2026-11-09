#
# Settings such as service URLs and credentials, read from the environment or a .env file.
#
# Copy env.default (in the repository root) to .env and fill it in. A variable set in the
# environment takes precedence over the same variable in .env.
#
import os

from dotenv import dotenv_values, find_dotenv


def get_setting(name: str, default: str | None = None) -> str | None:
    """
    Look up a setting from the environment, falling back to a .env file.

    The .env file is the nearest one found by searching upward from the current working
    directory. It is read on every call rather than loaded into os.environ: renci-ner is a
    library, so it shouldn't change the environment of the program importing it. Empty
    values (e.g. a line copied from env.default but not filled in) count as unset.

    :param name: The name of the environment variable.
    :param default: The value to return if the setting isn't set anywhere.
    :return: The value of the setting, or the default.
    """
    value = os.environ.get(name)
    if value:
        return value
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        value = dotenv_values(dotenv_path).get(name)
        if value:
            return value
    return default
