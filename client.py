"""
Test client for ComfyUI MCP Server using HTTP/JSON-RPC protocol.

This client connects to the MCP server using the streamable-http transport,
which uses standard HTTP requests with JSON-RPC protocol.
"""
import json
import sys
from typing import Any, Dict, Optional

import requests

# Configuration
MCP_ENDPOINT = "http://127.0.0.1:9000/mcp"
REQUEST_TIMEOUT = 300  # 5 minutes for long-running operations
REQUEST_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def parse_sse_response(response_text: str) -> dict:
    """Parse Server-Sent Events (SSE) response format."""
    lines = response_text.replace("\r\n", "\n").split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("data: "):
            json_str = line[6:]  # Remove "data: " prefix
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    raise ValueError("No valid JSON data found in SSE response")


def _make_request(method: str, params: Dict[str, Any], request_id: int = 1) -> Optional[dict]:
    """Make an MCP JSON-RPC request and return the parsed response."""
    request = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }

    try:
        response = requests.post(
            MCP_ENDPOINT,
            json=request,
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        # Handle both JSON and SSE responses
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            result = parse_sse_response(response.text)
        else:
            result = response.json()

        return result

    except requests.RequestException as e:
        print(f"Request error: {e}")
        if hasattr(e, "response") and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"Error details: {json.dumps(error_detail, indent=2)}")
            except Exception:
                print(f"Response text: {e.response.text[:500]}")
        return None

    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error parsing response: {e}")
        if hasattr(e, "response"):
            print(f"Response text: {response.text[:500]}")
        return None


def list_available_tools() -> list:
    """List all available tools from the MCP server."""
    result = _make_request("tools/list", {})
    if not result:
        return []

    if "result" in result and "tools" in result["result"]:
        tools = result["result"]["tools"]
        print(f"\nAvailable tools ({len(tools)}):")
        for tool in tools:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "No description")
            print(f"  • {name}: {desc}")
        return tools

    print("Unexpected response format:")
    print(json.dumps(result, indent=2))
    return []


def call_tool(tool_name: str, arguments: Dict[str, Any]) -> Optional[dict]:
    """Call an MCP tool with the given arguments."""
    print(f"\n📞 Calling tool '{tool_name}'")
    print(f"Arguments: {json.dumps(arguments, indent=2)}")

    result = _make_request("tools/call", {"name": tool_name, "arguments": arguments}, request_id=2)
    if not result:
        return None

    if "error" in result:
        print(f"\n❌ Error from server:")
        print(json.dumps(result["error"], indent=2))
        return None

    if "result" in result:
        print(f"\n✅ Response from server:")
        print(json.dumps(result["result"], indent=2))
        return result["result"]

    print("\n⚠️  Unexpected response format:")
    print(json.dumps(result, indent=2))
    return None


def print_section(title: str, width: int = 60):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def test_generate_image():
    """Test the generate_image tool (if available)."""
    print_section("ComfyUI MCP Server Test Client")

    # List available tools
    print("\n1️⃣  Listing available tools...")
    tools = list_available_tools()

    if not tools:
        print("\n❌ No tools available. Make sure the server is running and workflows are loaded.")
        return

    # Find generate_image tool or use first available
    tool_name = None
    for tool in tools:
        if tool.get("name") == "generate_image":
            tool_name = "generate_image"
            break

    if not tool_name and tools:
        tool_name = tools[0].get("name")
        print(f"\n⚠️  Using first available tool '{tool_name}' instead of 'generate_image'")

    if not tool_name:
        print("\n❌ No tools found to test.")
        return

    # Call the tool
    print(f"\n2️⃣  Testing tool '{tool_name}'...")
    arguments = {
        "prompt": "an english mastiff dog, mouth closed, standing majestically atop a large boulder, bright shiny day, forest background",
        "width": 512,
        "height": 512,
    }

    result = call_tool(tool_name, arguments)

    # Display results
    print_section("Test Results")
    if result:
        print("✅ Test completed successfully!")
        if "asset_url" in result:
            print(f"\n📎 Generated asset URL: {result['asset_url']}")
        elif "image_url" in result:
            print(f"\n📎 Generated image URL: {result['image_url']}")
    else:
        print("❌ Test failed. Check the error messages above.")


def main():
    """Main entry point."""
    try:
        test_generate_image()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
