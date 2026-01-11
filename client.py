"""
Test client for ComfyUI MCP Server using HTTP/JSON-RPC protocol.

This client connects to the MCP server using the streamable-http transport,
which uses standard HTTP requests with JSON-RPC protocol.
"""
import requests
import json
import sys

# MCP server endpoint
MCP_ENDPOINT = "http://127.0.0.1:9000/mcp"

def parse_sse_response(response_text: str) -> dict:
    """Parse Server-Sent Events (SSE) response format."""
    # SSE format: "event: message\r\ndata: {json}\r\n\r\n"
    # Handle both \n and \r\n line endings
    lines = response_text.replace('\r\n', '\n').split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('data: '):
            json_str = line[6:]  # Remove "data: " prefix
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                # Try to find the actual JSON if there's extra text
                continue
    raise ValueError("No valid JSON data found in SSE response")

def list_available_tools():
    """List all available tools from the MCP server."""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {}
    }
    
    try:
        # Streamable-http requires Accept header to include both content types
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        response = requests.post(MCP_ENDPOINT, json=request, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Handle both JSON and SSE responses
        content_type = response.headers.get('content-type', '')
        if 'text/event-stream' in content_type:
            result = parse_sse_response(response.text)
        else:
            result = response.json()
        
        if "result" in result and "tools" in result["result"]:
            tools = result["result"]["tools"]
            print(f"Available tools ({len(tools)}):")
            for tool in tools:
                print(f"  - {tool.get('name', 'unknown')}: {tool.get('description', 'No description')}")
            return tools
        else:
            print("Unexpected response format:")
            print(json.dumps(result, indent=2))
            return []
    except requests.RequestException as e:
        print(f"Error listing tools: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text[:500]}")
        return []
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error parsing response: {e}")
        print(f"Response text: {response.text[:500]}")
        return []

def call_tool(tool_name: str, arguments: dict):
    """Call an MCP tool with the given arguments."""
    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }
    
    try:
        print(f"\nCalling tool '{tool_name}' with arguments:")
        print(json.dumps(arguments, indent=2))
        print("\nSending request to MCP server...")
        
        # Streamable-http requires Accept header to include both content types
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        response = requests.post(MCP_ENDPOINT, json=request, headers=headers, timeout=300)  # 5 min timeout for image generation
        response.raise_for_status()
        
        # Handle both JSON and SSE responses
        content_type = response.headers.get('content-type', '')
        if 'text/event-stream' in content_type:
            result = parse_sse_response(response.text)
        else:
            result = response.json()
        
        if "error" in result:
            print(f"\nError from server:")
            print(json.dumps(result["error"], indent=2))
            return None
        
        if "result" in result:
            print(f"\nResponse from server:")
            print(json.dumps(result["result"], indent=2))
            return result["result"]
        else:
            print("\nUnexpected response format:")
            print(json.dumps(result, indent=2))
            return None
            
    except requests.RequestException as e:
        print(f"\nRequest error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print("Error details:")
                print(json.dumps(error_detail, indent=2))
            except:
                print(f"Response text: {e.response.text}")
        return None

def test_generate_image():
    """Test the generate_image tool (if available)."""
    print("=" * 60)
    print("ComfyUI MCP Server Test Client")
    print("=" * 60)
    
    # First, list available tools
    print("\n1. Listing available tools...")
    tools = list_available_tools()
    
    if not tools:
        print("\nNo tools available. Make sure the server is running and workflows are loaded.")
        return
    
    # Find generate_image tool or use first available tool
    tool_name = None
    for tool in tools:
        if tool.get("name") == "generate_image":
            tool_name = "generate_image"
            break
    
    if not tool_name and tools:
        tool_name = tools[0].get("name")
        print(f"\nNote: Using first available tool '{tool_name}' instead of 'generate_image'")
    
    if not tool_name:
        print("\nNo tools found to test.")
        return
    
    # Call the tool with test parameters
    print(f"\n2. Testing tool '{tool_name}'...")
    arguments = {
        "prompt": "an english mastiff dog sitting on a large boulder, bright shiny day",
        "width": 512,
        "height": 512
    }
    
    result = call_tool(tool_name, arguments)
    
    if result:
        print("\n" + "=" * 60)
        print("Test completed successfully!")
        if "asset_url" in result:
            print(f"\nGenerated asset URL: {result['asset_url']}")
        elif "image_url" in result:
            print(f"\nGenerated image URL: {result['image_url']}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Test failed. Check the error messages above.")
        print("=" * 60)

if __name__ == "__main__":
    try:
        test_generate_image()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
