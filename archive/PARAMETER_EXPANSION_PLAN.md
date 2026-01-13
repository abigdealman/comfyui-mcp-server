# Plan: Expanding generate_image Tool Parameters

## Current State

**Currently Exposed:**
- ✅ `prompt` (required, string) - Node 6, input "text"
- ✅ `seed` (optional, int) - Node 3, input "seed"

**Currently Hardcoded:**
- ❌ `width` = 512 - Node 5, input "width"
- ❌ `height` = 512 - Node 5, input "height"
- ❌ `model` = "v1-5-pruned-emaonly.ckpt" - Node 4, input "ckpt_name"
- ❌ `steps` = 20 - Node 3, input "steps"
- ❌ `cfg` = 8 - Node 3, input "cfg"
- ❌ `sampler_name` = "euler" - Node 3, input "sampler_name"
- ❌ `scheduler` = "normal" - Node 3, input "scheduler"
- ❌ `denoise` = 1 - Node 3, input "denoise"
- ❌ `negative_prompt` = "text, watermark" - Node 7, input "text"

## Solution: Add PARAM_ Placeholders

The infrastructure already supports this! Just add `PARAM_*` placeholders to the workflow JSON.

### Step 1: Update `workflows/generate_image.json`

Replace hardcoded values with placeholders:

```json
{
  "3": {
    "inputs": {
      "seed": "PARAM_INT_SEED",
      "steps": "PARAM_INT_STEPS",        // NEW
      "cfg": "PARAM_FLOAT_CFG",          // NEW
      "sampler_name": "PARAM_STR_SAMPLER_NAME",  // NEW
      "scheduler": "PARAM_STR_SCHEDULER",        // NEW
      "denoise": "PARAM_FLOAT_DENOISE",   // NEW
      ...
    }
  },
  "4": {
    "inputs": {
      "ckpt_name": "PARAM_MODEL"         // NEW
    }
  },
  "5": {
    "inputs": {
      "width": "PARAM_INT_WIDTH",        // NEW
      "height": "PARAM_INT_HEIGHT",      // NEW
      "batch_size": 1
    }
  },
  "7": {
    "inputs": {
      "text": "PARAM_NEGATIVE_PROMPT",    // NEW
      ...
    }
  }
}
```

### Step 2: Add Descriptions (Optional but Recommended)

Update `PLACEHOLDER_DESCRIPTIONS` in `server.py`:

```python
PLACEHOLDER_DESCRIPTIONS = {
    "prompt": "Main text prompt used inside the workflow.",
    "seed": "Random seed for image generation. If not provided, a random seed will be generated.",
    "width": "Image width in pixels. Default: 512.",
    "height": "Image height in pixels. Default: 512.",
    "model": "Checkpoint model name (e.g., 'v1-5-pruned-emaonly.ckpt', 'sd_xl_base_1.0.safetensors').",
    "steps": "Number of sampling steps. Higher = better quality but slower. Default: 20.",
    "cfg": "Classifier-free guidance scale. Higher = more adherence to prompt. Default: 8.",
    "sampler_name": "Sampling method (e.g., 'euler', 'dpmpp_2m', 'ddim'). Default: 'euler'.",
    "scheduler": "Scheduler type (e.g., 'normal', 'karras', 'exponential'). Default: 'normal'.",
    "denoise": "Denoising strength (0.0-1.0). Default: 1.0.",
    "negative_prompt": "Negative prompt to avoid certain elements. Default: 'text, watermark'.",
    "tags": "Comma-separated descriptive tags for the audio model.",
    "lyrics": "Full lyric text that should drive the audio generation.",
}
```

### Step 3: Decide Required vs Optional Parameters

**Recommended:**
- **Required:** `prompt` (already required)
- **Optional with defaults:** Everything else

This allows the tool to work with minimal parameters but gives full control when needed.

### Step 4: Handle Default Values

Two approaches:

#### Option A: Set defaults in workflow JSON (Recommended)
Keep current values as placeholders that get replaced only if provided:

```json
{
  "5": {
    "inputs": {
      "width": "PARAM_INT_WIDTH",   // If not provided, keep as string "PARAM_INT_WIDTH"?
      "height": "PARAM_INT_HEIGHT"
    }
  }
}
```

**Problem:** If parameter not provided, placeholder string remains in workflow.

#### Option B: Set defaults in render_workflow (Better)
Modify `render_workflow` to inject defaults for optional parameters:

```python
def render_workflow(self, definition: WorkflowToolDefinition, provided_params: Dict[str, Any]):
    workflow = copy.deepcopy(definition.template)
    
    # Default values for optional parameters
    defaults = {
        "width": 512,
        "height": 512,
        "steps": 20,
        "cfg": 8.0,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": 1.0,
        "model": "v1-5-pruned-emaonly.ckpt",
        "negative_prompt": "text, watermark",
    }
    
    for param in definition.parameters.values():
        if param.required and param.name not in provided_params:
            raise ValueError(f"Missing required parameter '{param.name}'")
        
        # Use provided value or default
        raw_value = provided_params.get(param.name)
        if raw_value is None:
            if param.name in defaults:
                raw_value = defaults[param.name]
                logger.debug(f"Using default value for {param.name}: {raw_value}")
            elif param.name == "seed":
                # Special handling for seed - generate random
                import random
                raw_value = random.randint(0, 2**32 - 1)
            else:
                continue
        
        coerced_value = self._coerce_value(raw_value, param.annotation)
        for node_id, input_name in param.bindings:
            workflow[node_id]["inputs"][input_name] = coerced_value
    
    return workflow
```

## Implementation Steps

1. **Update workflow JSON** - Add PARAM_ placeholders for desired parameters
2. **Add descriptions** - Update PLACEHOLDER_DESCRIPTIONS dictionary
3. **Add default handling** - Modify render_workflow to inject defaults
4. **Test** - Verify tool works with and without optional parameters

## Example: Before vs After

### Before (Current)
```python
# Tool signature:
generate_image(prompt: str, seed: Optional[int] = None)

# Usage:
generate_image(prompt="a cat")
```

### After (Proposed)
```python
# Tool signature:
generate_image(
    prompt: str,
    seed: Optional[int] = None,
    width: Optional[int] = None,      # defaults to 512
    height: Optional[int] = None,     # defaults to 512
    model: Optional[str] = None,      # defaults to "v1-5-pruned-emaonly.ckpt"
    steps: Optional[int] = None,      # defaults to 20
    cfg: Optional[float] = None,     # defaults to 8.0
    sampler_name: Optional[str] = None,  # defaults to "euler"
    scheduler: Optional[str] = None,     # defaults to "normal"
    denoise: Optional[float] = None,     # defaults to 1.0
    negative_prompt: Optional[str] = None # defaults to "text, watermark"
)

# Usage examples:
generate_image(prompt="a cat")  # Uses all defaults
generate_image(prompt="a cat", width=1024, height=768)  # Custom size
generate_image(prompt="a cat", model="sd_xl_base_1.0.safetensors", steps=30)  # Custom model & steps
```

## Benefits

1. **Backward Compatible** - Existing calls still work (all new params optional)
2. **Flexible** - Full control over image generation parameters
3. **No Code Changes Needed** - Just update workflow JSON and add defaults
4. **Auto-Discovery** - Parameters automatically appear in tool signature
5. **Type Safe** - Proper type annotations (int, float, str)

## Considerations

1. **Model Validation** - Should validate model name against available models?
2. **Parameter Ranges** - Should validate ranges (e.g., steps > 0, cfg > 0)?
3. **Sampler Options** - Should sampler_name be an enum of valid options?
4. **Performance** - More parameters = more complex tool signature, but MCP handles it well

## Recommendation

**Start Simple:**
1. Add `width`, `height`, `model`, `steps`, `cfg` as optional parameters
2. Add default handling in `render_workflow`
3. Add descriptions
4. Test thoroughly
5. Add more advanced parameters (sampler_name, scheduler, denoise) later if needed

This gives you the most commonly needed parameters without overwhelming the tool signature.
