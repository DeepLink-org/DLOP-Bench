_count = 0


def count():
    return _count


def reset_count():
    global _count
    _count = 0


def count_plus_one():
    global _count
    _count = _count + 1
