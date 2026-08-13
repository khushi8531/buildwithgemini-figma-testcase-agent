# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import contextlib
import os
from collections.abc import AsyncIterator

import google.auth
from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner
from google.cloud import logging as google_cloud_logging
from google.genai import types

from app.app_utils import services
from app.app_utils.a2a import attach_a2a_routes
from app.app_utils.reasoning_engine_adapter import (
    attach_reasoning_engine_routes,
)
from app.app_utils.typing import Feedback

load_dotenv()
otel_to_cloud = os.environ.get(
    "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY", ""
).lower() in ("true", "1")
_, project_id = google.auth.default()
logging_client = google_cloud_logging.Client()
logger = logging_client.logger(__name__)

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from app.agent import app as adk_app
    from app.agent import root_agent

    runner = Runner(
        app=adk_app,
        session_service=services.get_session_service(),
        artifact_service=services.get_artifact_service(),
        auto_create_session=True,
    )
    app.state.runner = runner
    app.state.agent_app_name = adk_app.name
    await attach_a2a_routes(
        app,
        agent=root_agent,
        runner=runner,
        task_store=InMemoryTaskStore(),
        rpc_path=f"/a2a/{adk_app.name}",
    )
    yield


app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=services.ARTIFACT_SERVICE_URI,
    allow_origins=["*"],
    session_service_uri=services.SESSION_SERVICE_URI,
    otel_to_cloud=otel_to_cloud,
    lifespan=lifespan,
)
app.title = "figma-testcase-agent"
app.description = "API for interacting with the Agent figma-testcase-agent"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

attach_reasoning_engine_routes(app)


# Mount static HTML UI
static_dir = os.path.join(AGENT_DIR, "app", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
@app.get("/portal", response_class=HTMLResponse)
async def get_portal():
    """Serves the interactive Zephyr & Jira test case generator UI."""
    html_path = os.path.join(AGENT_DIR, "app", "static", "ui.html")
    if os.path.exists(html_path):
        with open(html_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>UI portal file not found</h1>")


@app.post("/generate-testcases")
async def generate_testcases_endpoint(request: Request):
    """Endpoint handling UI inputs, optional image uploads, and executing agent pipeline to generate Zephyr CSV."""
    body = await request.json()
    jira_key = body.get("jira_key", "")
    confluence_id = body.get("confluence_id", "")
    figma_key = body.get("figma_key", "")
    figma_nodes = body.get("figma_nodes", "")
    fmt = body.get("format", "zephyr")
    image_data = body.get("image_data", "")

    parts = []

    if image_data and "," in image_data:
        header, encoded = image_data.split(",", 1)
        mime_type = "image/png"
        if "image/jpeg" in header or "image/jpg" in header:
            mime_type = "image/jpeg"
        elif "image/webp" in header:
            mime_type = "image/webp"

        try:
            img_bytes = base64.b64decode(encoded)
            parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
        except Exception as e:
            logger.warning(f"Error decoding base64 image: {e}")

    prompt_text = f"""Directly generate and return the complete test cases in {fmt.upper()} CSV format!
Inputs provided:
- Jira Ticket Key: {jira_key or 'PROJ-101'}
- Confluence Page ID: {confluence_id or '123456'}
- Figma File Key: {figma_key or 'aBc123XyZ'}
- Figma Node IDs: {figma_nodes or '1:2'}

Instructions:
1. If an image mockup is attached, analyze its UI fields, buttons, and state workflows visually.
2. Directly call format_{fmt}_test_cases (or format_zephyr_test_cases) or output the complete CSV table directly."""

    parts.append(types.Part.from_text(text=prompt_text))

    content_msg = types.Content(role="user", parts=parts)

    runner = request.app.state.runner
    response_text = ""
    async for event in runner.run_async(
        user_id="ui_user",
        session_id="ui_session",
        new_message=content_msg,
    ):
        if hasattr(event, "content") and event.content:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    response_text += part.text

    return JSONResponse({"status": "success", "csv": response_text})


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    logger.log_struct(feedback.model_dump(), severity="INFO")
    return {"status": "success"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
