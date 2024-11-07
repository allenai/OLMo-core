_MAJOR = "1"
_MINOR = "6"
_PATCH = "1"
_SUFFIX = ""

VERSION_SHORT = "{0}.{1}".format(_MAJOR, _MINOR)
VERSION = "{0}.{1}.{2}{3}".format(_MAJOR, _MINOR, _PATCH, _SUFFIX)


if __name__ == "__main__":
    import sys

    if sys.argv[-1] == "short":
        print(VERSION_SHORT)
    else:
        print(VERSION)
