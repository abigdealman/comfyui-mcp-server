# MCP Configuration for Cursor

This file explains how to configure Cursor to connect to the ComfyUI MCP Server.

## Configuration File

The `mcp.json` file is configured for Cursor's MCP integration. 

### Current Configuration

```json
{
  "mcpServers": {
    "comfyui": {
      "command": "python",
      "args": [
        "E:\\dev\\comfyui-mcp-server\\server.py",
        "--stdio"
      ],
      "env": {
        "COMFYUI_URL": "http://localhost:8188"
      }
    }
  }
}
```

### Important Notes

1. **Update the Path**: Make sure the path in `args` points to your actual `server.py` location:
   - Windows: `"E:\\dev\\comfyui-mcp-server\\server.py"`
   - Mac/Linux: `"/path/to/comfyui-mcp-server/server.py"`

2. **ComfyUI URL**: The `COMFYUI_URL` environment variable should point to your ComfyUI instance (default: `http://localhost:8188`)

3. **Transport**: The `--stdio` argument tells the server to use stdio transport, which is required for command-based MCP connections in Cursor.

## How to Use in Cursor

1. **Start the MCP Server** (required first step):
   ```bash
   python server.py
   ```
   The server will start on `http://127.0.0.1:9000/mcp`

2. **Locate Cursor's MCP Config**: 
   - The MCP configuration file location varies by platform
   - Check Cursor's settings or documentation for the exact location

3. **Add the Configuration**:
   - Either merge the `mcpServers` section into your existing MCP config
   - Or replace your MCP config with the contents of `mcp.json`

4. **Restart Cursor**: After updating the configuration, restart Cursor to load the new MCP server

5. **Verify Connection**: 
   - Cursor should show the ComfyUI MCP server as available
   - You should see tools like `generate_image` and `generate_song` available

## Available Tools

Once connected, you'll have access to:

- **generate_image**: Generate images using ComfyUI
  - Parameters: `prompt` (required), `width`, `height`, etc.
  
- **generate_song**: Generate audio/songs using ComfyUI
  - Parameters: `tags` (required), `lyrics` (required)

## Troubleshooting

### Server Not Connecting

1. **Check Python Path**: Make sure `python` in the command is the correct Python interpreter
   - You might need to use `python3` on Mac/Linux
   - Or use the full path: `"C:\\Python\\python.exe"`

2. **Check Server Path**: Verify the path to `server.py` is correct and absolute

3. **Check Dependencies**: Ensure all Python dependencies are installed:
   ```bash
   pip install requests mcp
   ```

4. **Check ComfyUI**: Make sure ComfyUI is running on the configured port (default: 8188)

### Tools Not Appearing

1. **Check Workflows**: Ensure workflow files exist in the `workflows/` directory
2. **Check Logs**: Look at Cursor's logs or server output for errors
3. **Verify Workflow Format**: Workflows must contain `PARAM_*` placeholders to be auto-discovered

## Alternative: Command-based Connection (stdio)

If you prefer Cursor to manage the server process automatically, you can use command-based connection:

```json
{
  "mcpServers": {
    "comfyui": {
      "command": "python",
      "args": [
        "E:\\dev\\comfyui-mcp-server\\server.py",
        "--stdio"
      ],
      "env": {
        "COMFYUI_URL": "http://localhost:8188"
      }
    }
  }
}
```

This will have Cursor start the server automatically using stdio transport. However, the HTTP-based connection (current config) is recommended as it allows the server to run independently and be accessed by multiple clients.
