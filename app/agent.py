# ruff: noqa
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

import csv
import io
import json
import os
import subprocess
from typing import Any

import requests
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types


MODEL = "gemini-3.6-flash"


def format_zephyr_test_cases(test_cases: list[dict[str, Any]]) -> str:
    """Formats a list of test cases into a valid Zephyr (Zephyr Scale / Squad) CSV structure for bulk import.

    Args:
        test_cases: A list of test case dictionaries containing:
            - name: Short title/summary of the test case
            - objective: Objective or description of the test
            - precondition: Required initial setup
            - folder: Folder path in Zephyr (e.g. "/UI/Login")
            - priority: Priority (High, Normal, Low)
            - component: Associated component (e.g. "Checkout UI")
            - step_action: Action steps to perform
            - step_data: Test input data
            - step_result: Expected result for the step

    Returns:
        CSV string in official Zephyr bulk import format.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write official Zephyr CSV headers
    writer.writerow([
        "Name",
        "Objective",
        "Precondition",
        "Folder",
        "Status",
        "Priority",
        "Component",
        "Owner",
        "Estimated Time",
        "Step Action",
        "Step Data",
        "Step Expected Result",
    ])
    
    for tc in test_cases:
        writer.writerow([
            tc.get("name") or tc.get("summary") or tc.get("title", ""),
            tc.get("objective") or tc.get("description", ""),
            tc.get("precondition") or tc.get("preconditions", ""),
            tc.get("folder", "/Automated Agent Generation"),
            tc.get("status", "Draft"),
            tc.get("priority", "Normal"),
            tc.get("component", "General"),
            tc.get("owner", ""),
            tc.get("estimated_time", ""),
            tc.get("step_action") or tc.get("steps", ""),
            tc.get("step_data") or tc.get("data", ""),
            tc.get("step_result") or tc.get("expected_result", ""),
        ])
        
    return output.getvalue()


def format_jira_xray_test_cases(test_cases: list[dict[str, Any]]) -> str:
    """Formats a list of test cases into a valid Jira / Xray CSV structure for bulk import."""
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow([
        "Issue Type",
        "Issue Key",
        "Summary",
        "Description",
        "Preconditions",
        "Priority",
        "Action",
        "Data",
        "Expected Result",
    ])
    
    for tc in test_cases:
        writer.writerow([
            tc.get("issue_type", "Test"),
            tc.get("test_id", ""),
            tc.get("summary") or tc.get("name", ""),
            tc.get("description") or tc.get("objective", ""),
            tc.get("preconditions") or tc.get("precondition", ""),
            tc.get("priority", "Medium"),
            tc.get("step_action", ""),
            tc.get("step_data", ""),
            tc.get("expected_result") or tc.get("step_result", ""),
        ])
        
    return output.getvalue()


def fetch_jira_ticket(jira_key: str, domain: str = "", auth_token: str = "") -> str:
    """Fetches details of a Jira ticket via Jira REST API using provided credentials or gcloud auth token.

    Args:
        jira_key: Jira issue key (e.g., "PROJ-123")
        domain: Atlassian domain name (e.g., "company" for company.atlassian.net)
        auth_token: Optional API token / OAuth token.

    Returns:
        JSON string or formatted summary of the Jira ticket.
    """
    if not domain:
        domain = os.environ.get("JIRA_DOMAIN", "")
    
    if not auth_token:
        auth_token = os.environ.get("JIRA_API_TOKEN", "") or os.environ.get("ATLASSIAN_TOKEN", "")

    if not domain:
        return f"Simulated Jira Ticket [{jira_key}]: Login & Checkout Specifications. Acceptance Criteria: Must validate required inputs, credit card length, and email format."

    url = f"https://{domain}.atlassian.net/rest/api/3/issue/{jira_key}"
    headers = {"Accept": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            fields = data.get("fields", {})
            summary = fields.get("summary", "")
            desc = fields.get("description", "")
            return f"Jira Issue [{jira_key}]: Summary: {summary}\nDescription: {desc}"
        else:
            return f"Jira API returned status {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return f"Error connecting to Jira API: {str(e)}"


def fetch_confluence_page(page_id: str, domain: str = "", auth_token: str = "") -> str:
    """Fetches specification text from a Confluence page via Confluence REST API.

    Args:
        page_id: Confluence Page ID (e.g. "123456789")
        domain: Atlassian domain (e.g. "company" for company.atlassian.net)
        auth_token: Optional API / Bearer token.

    Returns:
        Page content text.
    """
    if not domain:
        domain = os.environ.get("CONFLUENCE_DOMAIN", "") or os.environ.get("JIRA_DOMAIN", "")
    if not auth_token:
        auth_token = os.environ.get("CONFLUENCE_API_TOKEN", "") or os.environ.get("ATLASSIAN_TOKEN", "")

    if not domain or not auth_token:
        return f"Simulated Confluence Page [{page_id}]: Requirement Specifications. 1. Submit button disabled when inputs empty. 2. Loading state on submit click. 3. Display error message on payment failure."

    url = f"https://{domain}.atlassian.net/wiki/rest/api/content/{page_id}?expand=body.storage"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {auth_token}"}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            title = data.get("title", "")
            body = data.get("body", {}).get("storage", {}).get("value", "")
            return f"Confluence Page [{page_id}] Title: {title}\nContent:\n{body}"
        else:
            return f"Confluence API returned status {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return f"Error connecting to Confluence API: {str(e)}"


def fetch_figma_nodes(file_key: str, node_ids: str = "", figma_token: str = "") -> str:
    """Fetches Figma node structures, layer hierarchy, text content, and component properties via Figma REST API.

    Args:
        file_key: Figma file key from URL (e.g. "aBc123XyZ")
        node_ids: Comma-separated list of node IDs (e.g. "1:2,3:4")
        figma_token: Figma Personal Access Token

    Returns:
        JSON/text summary of Figma nodes.
    """
    if not figma_token:
        figma_token = os.environ.get("FIGMA_ACCESS_TOKEN", "") or os.environ.get("FIGMA_TOKEN", "")

    if not figma_token:
        return f"Simulated Figma File [{file_key}] Nodes [{node_ids}]: Frame 'Checkout Page' containing: Username Input, Password Input, Submit Button (Blue, 44px height), Error Message Banner (Red)."

    url = f"https://api.figma.com/v1/files/{file_key}/nodes?ids={node_ids}" if node_ids else f"https://api.figma.com/v1/files/{file_key}"
    headers = {"X-Figma-Token": figma_token}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            return f"Figma Nodes Data: {json.dumps(resp.json())[:1500]}"
        else:
            return f"Figma API returned status {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return f"Error connecting to Figma API: {str(e)}"


AGENT_INSTRUCTION = """You are an expert QA Automation & Test Engineering Specialist AI Agent.
Your role is to analyze Jira tickets, Confluence PRD/specifications, Figma UI node structures, AND UI mockup/screenshot images to generate comprehensive, high-quality test cases formatted for Zephyr (Zephyr Scale / Squad) or Jira Xray bulk import.

When given Jira ticket keys, Confluence page IDs, Figma file keys, or Figma node IDs:
1. Use `fetch_jira_ticket`, `fetch_confluence_page`, and `fetch_figma_nodes` tools to retrieve real-time requirements, visual component details, and acceptance criteria.

When an image (UI mockup, wireframe, or screenshot) is provided:
1. Inspect the image visually to identify all input fields, buttons, labels, validation rules, navigation elements, and layout states.
2. Even if no Figma or Jira API credentials or keys are provided, accept the image and generate full test case scenarios!

Test Case Generation Rules:
1. Cross-reference visual components/images with functional rules to derive:
   - Happy Paths (core user flows)
   - Negative Tests & Validation Errors (missing inputs, invalid formats)
   - Edge Cases & Boundary Value Analysis
   - UI Layout & Visual States
2. Call `format_zephyr_test_cases` (or `format_jira_xray_test_cases` if requested) to produce the exact bulk-import CSV file.
3. Output the complete CSV ready for download/import, along with a summary of test coverage.
"""


root_agent = Agent(
    name="root_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=AGENT_INSTRUCTION,
    tools=[
        fetch_jira_ticket,
        fetch_confluence_page,
        fetch_figma_nodes,
        format_zephyr_test_cases,
        format_jira_xray_test_cases,
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
)
