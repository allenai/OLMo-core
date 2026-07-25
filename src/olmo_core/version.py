_MAJOR = "2"
_MINOR = "5"
_PATCH = "0"
_SUFFIX = ""

VERSION_SHORT = f"{_MAJOR}.{_MINOR}"
VERSION = f"{_MAJOR}.{_MINOR}.{_PATCH}{_SUFFIX}"


if __name__ == "__main__":
    import sys

    if sys.argv[-1] == "short":
        print(VERSION_SHORT)
    else:
        print(VERSION)
