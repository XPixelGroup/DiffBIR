try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False
    print("No module 'xformers'. Proceeding without it.")


def is_xformers_available() -> bool:
    global XFORMERS_IS_AVAILBLE
    return XFORMERS_IS_AVAILBLE

def disable_xformers() -> None:
    print("DISABLE XFORMERS!")
    global XFORMERS_IS_AVAILBLE
    XFORMERS_IS_AVAILBLE = False
