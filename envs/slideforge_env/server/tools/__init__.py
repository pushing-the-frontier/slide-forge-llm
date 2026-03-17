from .research import web_search, fetch_url
from .content import create_outline, revise_outline
from .design import generate_slide, edit_slide, set_theme
from .meta import review_deck, finalize
from .structure import get_slide_content, delete_slide, reorder_slides, duplicate_slide, insert_slide

TOOL_REGISTRY = {
    "web_search": web_search,
    "fetch_url": fetch_url,
    "create_outline": create_outline,
    "revise_outline": revise_outline,
    "generate_slide": generate_slide,
    "edit_slide": edit_slide,
    "set_theme": set_theme,
    "review_deck": review_deck,
    "finalize": finalize,
    "get_slide_content": get_slide_content,
    "delete_slide": delete_slide,
    "reorder_slides": reorder_slides,
    "duplicate_slide": duplicate_slide,
    "insert_slide": insert_slide,
}
