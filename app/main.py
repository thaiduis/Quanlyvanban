from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
import os
import re
from markupsafe import Markup
from typing import Union

from .database import init_db, ensure_fulltext_indexes

load_dotenv()

# Initialize FastAPI with middleware
middleware = [
    Middleware(
        SessionMiddleware,
        secret_key=os.getenv("SECRET_KEY", "your-secret-key-here"),
        session_cookie="session"
    )
]

app = FastAPI(
    title="Document Manager (OCR)",
    middleware=middleware
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
static_dir = os.path.join(PROJECT_DIR, "static")
templates_dir = os.path.join(PROJECT_DIR, "templates")
upload_dir = os.path.join(PROJECT_DIR, os.getenv("UPLOAD_DIR", "uploads"))

os.makedirs(upload_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Add flash message support to templates
def flash(request: Request, message: str, category: str = "info") -> None:
    if "flash_messages" not in request.session:
        request.session["flash_messages"] = []
    request.session["flash_messages"].append({"message": message, "category": category})

def get_flashed_messages(request: Request, with_categories: bool = False):
    messages = request.session.pop("flash_messages", [])
    if with_categories:
        return [(msg["category"], msg["message"]) for msg in messages]
    return [msg["message"] for msg in messages]

# Add template context processors
@app.middleware("http")
async def template_context(request: Request, call_next):
    # Add functions to templates
    request.state.flash = lambda msg, cat="info": flash(request, msg, cat)
    request.state.get_flashed_messages = lambda: get_flashed_messages(request, True)
    
    # Override template function with request-specific one
    def get_flashed_messages_with_request(with_categories=False):
        return get_flashed_messages(request, with_categories)
    
    # Store original function
    original_get_flashed_messages = templates.env.globals.get("get_flashed_messages")
    
    # Override for this request
    templates.env.globals["get_flashed_messages"] = get_flashed_messages_with_request
    
    response = await call_next(request)
    
    # Restore original function
    templates.env.globals["get_flashed_messages"] = original_get_flashed_messages
    
    return response

# Add highlight filter for templates
def highlight_filter(text: str, keyword: Union[str, None]) -> Union[Markup, str]:
    if not text or not keyword:
        return text or ""
    try:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        replaced = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)
        return Markup(replaced)
    except Exception:
        return text

templates.env.filters["highlight"] = highlight_filter

# Make flash functions available to templates
def get_flashed_messages_template(with_categories=False):
    """Template function to get flashed messages"""
    return []
templates.env.globals["get_flashed_messages"] = get_flashed_messages_template

# Add Python built-in functions to templates
templates.env.globals["max"] = max
templates.env.globals["min"] = min
templates.env.globals["range"] = range
templates.env.globals["len"] = len

@app.on_event("startup")
async def on_startup() -> None:
    try:
        init_db()
        ensure_fulltext_indexes()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Database initialization failed: {e}")
        # Continue without database for now

@app.get("/")
async def home(request: Request):
    return RedirectResponse(url="/documents")

@app.get("/test")
async def test():
    return {"message": "Server is working!", "status": "ok"}

# Import và include router sau khi app tạo để tránh vòng import
from .routers.documents import router as documents_router  # noqa: E402
app.include_router(documents_router, prefix="/documents", tags=["documents"])
