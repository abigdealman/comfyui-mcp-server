import copy
import inspect
import json
import logging
import os
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional, Sequence, Tuple

from mcp.server.fastmcp import FastMCP
from comfyui_client import ComfyUIClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MCP_Server")

PLACEHOLDER_PREFIX = "PARAM_"
PLACEHOLDER_TYPE_HINTS = {
    "STR": str,
    "STRING": str,
    "TEXT": str,
    "INT": int,
    "FLOAT": float,
    "BOOL": bool,
}
PLACEHOLDER_DESCRIPTIONS = {
    "prompt": "Main text prompt used inside the workflow.",
    "seed": "Random seed for image generation. If not provided, a random seed will be generated.",
    "width": "Image width in pixels. Default: 512.",
    "height": "Image height in pixels. Default: 512.",
    "model": "Checkpoint model name (e.g., 'v1-5-pruned-emaonly.ckpt', 'sd_xl_base_1.0.safetensors'). Default: 'v1-5-pruned-emaonly.ckpt'.",
    "steps": "Number of sampling steps. Higher = better quality but slower. Default: 20.",
    "cfg": "Classifier-free guidance scale. Higher = more adherence to prompt. Default: 8.0.",
    "sampler_name": "Sampling method (e.g., 'euler', 'dpmpp_2m', 'ddim'). Default: 'euler'.",
    "scheduler": "Scheduler type (e.g., 'normal', 'karras', 'exponential'). Default: 'normal'.",
    "denoise": "Denoising strength (0.0-1.0). Default: 1.0.",
    "negative_prompt": "Negative prompt to avoid certain elements. Default: 'text, watermark'.",
    "tags": "Comma-separated descriptive tags for the audio model.",
    "lyrics": "Full lyric text that should drive the audio generation.",
    "seconds": "Audio duration in seconds. Default: 60 (1 minute).",
    "lyrics_strength": "How strongly lyrics influence audio generation (0.0-1.0). Default: 0.99.",
    "duration": "Video duration in seconds. Default: 5.",
    "fps": "Frames per second for video output. Default: 16.",
}
DEFAULT_OUTPUT_KEYS = ("images", "image", "gifs", "gif")
AUDIO_OUTPUT_KEYS = ("audio", "audios", "sound", "files")
VIDEO_OUTPUT_KEYS = ("videos", "video", "mp4", "mov", "webm")

# Configuration paths
CONFIG_DIR = Path.home() / ".config" / "comfy-mcp"
CONFIG_FILE = CONFIG_DIR / "config.json"
WORKFLOW_DIR = Path(os.getenv("COMFY_MCP_WORKFLOW_DIR", str(Path(__file__).parent / "workflows")))


class DefaultsManager:
    """Manages default values with precedence: per-call > runtime > config > env > hardcoded"""
    
    def __init__(self, comfyui_client: ComfyUIClient):
        self.comfyui_client = comfyui_client
        self._runtime_defaults: Dict[str, Dict[str, Any]] = {
            "image": {},
            "audio": {},
            "video": {}
        }
        self._config_defaults = self._load_config_defaults()
        self._hardcoded_defaults = {
            "image": {
                "width": 512,
                "height": 512,
                "steps": 20,
                "cfg": 8.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": "v1-5-pruned-emaonly.ckpt",
                "negative_prompt": "text, watermark",
            },
            "audio": {
                "steps": 50,
                "cfg": 5.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "seconds": 60,
                "lyrics_strength": 0.99,
                "model": "ace_step_v1_3.5b.safetensors",
            },
            "video": {
                "width": 1280,
                "height": 720,
                "steps": 20,
                "cfg": 8.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": "wan2.2_vae.safetensors",
                "negative_prompt": "text, watermark",
                "duration": 5,
                "fps": 16,
            }
        }
    
    def _load_config_defaults(self) -> Dict[str, Dict[str, Any]]:
        """Load defaults from config file"""
        defaults = {"image": {}, "audio": {}, "video": {}}
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    defaults["image"] = config.get("defaults", {}).get("image", {})
                    defaults["audio"] = config.get("defaults", {}).get("audio", {})
                    defaults["video"] = config.get("defaults", {}).get("video", {})
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load config file {CONFIG_FILE}: {e}")
        return defaults
    
    def _get_env_defaults(self) -> Dict[str, Dict[str, Any]]:
        """Load defaults from environment variables"""
        defaults = {"image": {}, "audio": {}, "video": {}}
        image_model = os.getenv("COMFY_MCP_DEFAULT_IMAGE_MODEL")
        audio_model = os.getenv("COMFY_MCP_DEFAULT_AUDIO_MODEL")
        video_model = os.getenv("COMFY_MCP_DEFAULT_VIDEO_MODEL")
        if image_model:
            defaults["image"]["model"] = image_model
        if audio_model:
            defaults["audio"]["model"] = audio_model
        if video_model:
            defaults["video"]["model"] = video_model
        return defaults
    
    def get_default(self, namespace: str, key: str, provided_value: Any = None) -> Any:
        """Get default value with precedence: provided > runtime > config > env > hardcoded"""
        if provided_value is not None:
            return provided_value
        
        # Check runtime defaults (highest priority after provided)
        if key in self._runtime_defaults.get(namespace, {}):
            return self._runtime_defaults[namespace][key]
        
        # Check config file defaults
        if key in self._config_defaults.get(namespace, {}):
            return self._config_defaults[namespace][key]
        
        # Check environment variables
        env_defaults = self._get_env_defaults()
        if key in env_defaults.get(namespace, {}):
            return env_defaults[namespace][key]
        
        # Check hardcoded defaults (lowest priority)
        if key in self._hardcoded_defaults.get(namespace, {}):
            return self._hardcoded_defaults[namespace][key]
        
        return None
    
    def get_all_defaults(self) -> Dict[str, Dict[str, Any]]:
        """Get all effective defaults (merged from all sources)"""
        env_defaults = self._get_env_defaults()
        result = {
            "image": {},
            "audio": {},
            "video": {}
        }
        
        for namespace in ["image", "audio", "video"]:
            # Start with hardcoded
            result[namespace] = self._hardcoded_defaults[namespace].copy()
            # Override with env
            result[namespace].update(env_defaults.get(namespace, {}))
            # Override with config
            result[namespace].update(self._config_defaults.get(namespace, {}))
            # Override with runtime (highest)
            result[namespace].update(self._runtime_defaults.get(namespace, {}))
        
        return result
    
    def set_defaults(self, namespace: str, defaults: Dict[str, Any], validate_models: bool = True) -> Dict[str, Any]:
        """Set runtime defaults for a namespace. Returns validation errors if any."""
        errors = []
        
        if namespace not in ["image", "audio", "video"]:
            return {"error": f"Invalid namespace: {namespace}. Must be 'image', 'audio', or 'video'"}
        
        # Validate model names if provided
        if validate_models and "model" in defaults:
            model_name = defaults["model"]
            available_models = self.comfyui_client.available_models
            if available_models and model_name not in available_models:
                errors.append(f"Model '{model_name}' not found. Available models: {available_models[:5]}...")
        
        if errors:
            return {"errors": errors}
        
        # Update runtime defaults
        if namespace not in self._runtime_defaults:
            self._runtime_defaults[namespace] = {}
        self._runtime_defaults[namespace].update(defaults)
        
        return {"success": True, "updated": defaults}
    
    def persist_defaults(self, namespace: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Persist defaults to config file"""
        # Ensure config directory exists
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load existing config
        config = {}
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except (json.JSONDecodeError, IOError):
                config = {}
        
        # Update defaults
        if "defaults" not in config:
            config["defaults"] = {}
        if namespace not in config["defaults"]:
            config["defaults"][namespace] = {}
        config["defaults"][namespace].update(defaults)
        
        # Save config
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            # Reload config defaults
            self._config_defaults = self._load_config_defaults()
            return {"success": True, "persisted": defaults}
        except IOError as e:
            return {"error": f"Failed to write config file: {e}"}


@dataclass
class WorkflowParameter:
    name: str
    placeholder: str
    annotation: type
    description: str
    bindings: list[Tuple[str, str]] = field(default_factory=list)
    required: bool = True


@dataclass
class WorkflowToolDefinition:
    workflow_id: str
    tool_name: str
    description: str
    template: Dict[str, Any]
    parameters: "OrderedDict[str, WorkflowParameter]"
    output_preferences: Sequence[str]


class WorkflowManager:
    def __init__(self, workflows_dir: Path):
        self.workflows_dir = Path(workflows_dir).resolve()
        self._tool_names: set[str] = set()
        self.tool_definitions = self._load_workflows()
        self._workflow_cache: Dict[str, Dict[str, Any]] = {}
    
    def _safe_workflow_path(self, workflow_id: str) -> Optional[Path]:
        """Resolve workflow ID to file path with path traversal protection"""
        # Normalize workflow_id (remove any path separators and dangerous characters)
        safe_id = workflow_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        # Remove any remaining path-like characters
        safe_id = "".join(c for c in safe_id if c.isalnum() or c in ("_", "-"))
        if not safe_id:
            logger.warning(f"Invalid workflow_id after sanitization: {workflow_id}")
            return None
        
        workflow_path = (self.workflows_dir / f"{safe_id}.json").resolve()
        
        # Ensure the resolved path is within workflows_dir
        try:
            workflow_path.relative_to(self.workflows_dir.resolve())
        except ValueError:
            logger.warning(f"Path traversal attempt detected: {workflow_id}")
            return None
        
        return workflow_path if workflow_path.exists() else None
    
    def _load_workflow_metadata(self, workflow_path: Path) -> Dict[str, Any]:
        """Load sidecar metadata file if it exists"""
        metadata_path = workflow_path.with_suffix(".meta.json")
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load metadata for {workflow_path.name}: {e}")
        return {}
    
    def get_workflow_catalog(self) -> list[Dict[str, Any]]:
        """Get catalog of all available workflows"""
        catalog = []
        if not self.workflows_dir.exists():
            return catalog
        
        for workflow_path in sorted(self.workflows_dir.glob("*.json")):
            # Skip metadata files
            if workflow_path.name.endswith(".meta.json"):
                continue
            
            workflow_id = workflow_path.stem
            try:
                with open(workflow_path, "r", encoding="utf-8") as f:
                    workflow = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Skipping {workflow_path.name}: {e}")
                continue
            
            # Load metadata
            metadata = self._load_workflow_metadata(workflow_path)
            
            # Extract parameters
            parameters = self._extract_parameters(workflow)
            available_inputs = {
                name: {
                    "type": param.annotation.__name__,
                    "required": param.required,
                    "description": param.description
                }
                for name, param in parameters.items()
            }
            
            # Get workflow defaults from metadata or infer from workflow_id
            workflow_defaults = metadata.get("defaults", {})
            if not workflow_defaults and workflow_id in ["generate_image", "generate_song", "generate_video"]:
                # Use namespace-based defaults
                namespace = self._determine_namespace(workflow_id)
                # This will be populated by defaults_manager when needed
            
            catalog.append({
                "id": workflow_id,
                "name": metadata.get("name", workflow_id.replace("_", " ").title()),
                "description": metadata.get("description", f"Execute the '{workflow_id}' workflow."),
                "available_inputs": available_inputs,
                "defaults": workflow_defaults,
                "updated_at": metadata.get("updated_at"),
                "hash": metadata.get("hash"),  # Could compute file hash if needed
            })
        
        return catalog
    
    def load_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Load workflow by ID with caching"""
        if workflow_id in self._workflow_cache:
            return copy.deepcopy(self._workflow_cache[workflow_id])
        
        workflow_path = self._safe_workflow_path(workflow_id)
        if not workflow_path:
            return None
        
        try:
            with open(workflow_path, "r", encoding="utf-8") as f:
                workflow = json.load(f)
            self._workflow_cache[workflow_id] = workflow
            return copy.deepcopy(workflow)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load workflow {workflow_id}: {e}")
            return None
    
    def apply_workflow_overrides(self, workflow: Dict[str, Any], workflow_id: str, overrides: Dict[str, Any], defaults_manager: Optional["DefaultsManager"] = None) -> Dict[str, Any]:
        """Apply constrained overrides to workflow based on metadata"""
        workflow_path = self._safe_workflow_path(workflow_id)
        if not workflow_path:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        metadata = self._load_workflow_metadata(workflow_path)
        override_mappings = metadata.get("override_mappings", {})
        constraints = metadata.get("constraints", {})
        
        # If no metadata, try to infer from PARAM_ placeholders
        if not override_mappings:
            parameters = self._extract_parameters(workflow)
            for param_name, param in parameters.items():
                # Build mapping from parameter name to node bindings
                if param_name not in override_mappings:
                    override_mappings[param_name] = param.bindings
        
        # Determine namespace for defaults
        namespace = self._determine_namespace(workflow_id)
        
        # Apply overrides with constraints
        for param_name, value in overrides.items():
            if param_name not in override_mappings:
                logger.warning(f"Override '{param_name}' not in declared mappings for {workflow_id}, skipping")
                continue
            
            # Apply constraints if defined
            if param_name in constraints:
                constraint = constraints[param_name]
                if "enum" in constraint and value not in constraint["enum"]:
                    raise ValueError(f"Value '{value}' for '{param_name}' not in allowed enum: {constraint['enum']}")
                if "min" in constraint and value < constraint["min"]:
                    raise ValueError(f"Value '{value}' for '{param_name}' below minimum: {constraint['min']}")
                if "max" in constraint and value > constraint["max"]:
                    raise ValueError(f"Value '{value}' for '{param_name}' above maximum: {constraint['max']}")
            
            # Get parameter type from extracted parameters
            parameters = self._extract_parameters(workflow)
            if param_name in parameters:
                param = parameters[param_name]
                coerced_value = self._coerce_value(value, param.annotation)
            else:
                coerced_value = value
            
            # Apply to all bindings
            for node_id, input_name in override_mappings[param_name]:
                if node_id in workflow and "inputs" in workflow[node_id]:
                    workflow[node_id]["inputs"][input_name] = coerced_value
        
        # Apply defaults for parameters not in overrides
        parameters = self._extract_parameters(workflow)
        for param_name, param in parameters.items():
            if param_name not in overrides and not param.required:
                if defaults_manager:
                    default_value = defaults_manager.get_default(namespace, param.name, None)
                    if default_value is not None:
                        for node_id, input_name in param.bindings:
                            if node_id in workflow and "inputs" in workflow[node_id]:
                                workflow[node_id]["inputs"][input_name] = default_value
        
        return workflow

    def _load_workflows(self):
        definitions: list[WorkflowToolDefinition] = []
        if not self.workflows_dir.exists():
            logger.info("Workflow directory %s does not exist yet", self.workflows_dir)
            return definitions

        for workflow_path in sorted(self.workflows_dir.glob("*.json")):
            try:
                with open(workflow_path, "r", encoding="utf-8") as handle:
                    workflow = json.load(handle)
            except json.JSONDecodeError as exc:
                logger.error("Skipping workflow %s due to JSON error: %s", workflow_path.name, exc)
                continue

            parameters = self._extract_parameters(workflow)
            if not parameters:
                logger.info(
                    "Workflow %s has no %s placeholders; skipping auto-tool registration",
                    workflow_path.name,
                    PLACEHOLDER_PREFIX,
                )
                continue

            tool_name = self._dedupe_tool_name(self._derive_tool_name(workflow_path.stem))
            definition = WorkflowToolDefinition(
                workflow_id=workflow_path.stem,
                tool_name=tool_name,
                description=self._derive_description(workflow_path.stem),
                template=workflow,
                parameters=parameters,
                output_preferences=self._guess_output_preferences(workflow),
            )
            logger.info(
                "Prepared workflow tool '%s' from %s with params %s",
                tool_name,
                workflow_path.name,
                list(parameters.keys()),
            )
            definitions.append(definition)

        return definitions

    def render_workflow(self, definition: WorkflowToolDefinition, provided_params: Dict[str, Any], defaults_manager: Optional["DefaultsManager"] = None):
        workflow = copy.deepcopy(definition.template)
        
        # Determine namespace (image, audio, or video)
        namespace = self._determine_namespace(definition.workflow_id)
        
        for param in definition.parameters.values():
            if param.required and param.name not in provided_params:
                raise ValueError(f"Missing required parameter '{param.name}'")
            
            # Use provided value, default, or generate (for seed)
            raw_value = provided_params.get(param.name)
            if raw_value is None:
                if param.name == "seed" and param.annotation is int:
                    # Special handling for seed - generate random
                    import random
                    raw_value = random.randint(0, 2**32 - 1)
                    logger.debug(f"Generated random seed: {raw_value}")
                elif defaults_manager:
                    # Use defaults manager to get value with proper precedence
                    raw_value = defaults_manager.get_default(namespace, param.name, None)
                    if raw_value is not None:
                        logger.debug(f"Using default value for {param.name}: {raw_value}")
                    else:
                        # Skip parameters without defaults
                        continue
                else:
                    # Fallback to old behavior if no defaults manager
                    continue
            
            coerced_value = self._coerce_value(raw_value, param.annotation)
            for node_id, input_name in param.bindings:
                workflow[node_id]["inputs"][input_name] = coerced_value
        
        return workflow

    def _extract_parameters(self, workflow: Dict[str, Any]):
        parameters: "OrderedDict[str, WorkflowParameter]" = OrderedDict()
        for node_id, node in workflow.items():
            inputs = node.get("inputs", {})
            if not isinstance(inputs, dict):
                continue
            for input_name, value in inputs.items():
                parsed = self._parse_placeholder(value)
                if not parsed:
                    continue
                param_name, annotation, placeholder_value = parsed
                description = PLACEHOLDER_DESCRIPTIONS.get(
                    param_name, f"Value for '{param_name}'."
                )
                parameter = parameters.get(param_name)
                if not parameter:
                    # Make seed and other optional parameters non-required
                    # Only 'prompt' should be required for generate_image
                    # Only 'tags' and 'lyrics' should be required for generate_song
                    # Only 'prompt' should be required for generate_video
                    optional_params = {
                        "seed", "width", "height", "model", "steps", "cfg",
                        "sampler_name", "scheduler", "denoise", "negative_prompt",
                        "seconds", "lyrics_strength",  # Audio-specific optional params
                        "duration", "fps"  # Video-specific optional params
                    }
                    is_required = param_name not in optional_params
                    parameter = WorkflowParameter(
                        name=param_name,
                        placeholder=placeholder_value,
                        annotation=annotation,
                        description=description,
                        required=is_required,
                    )
                    parameters[param_name] = parameter
                parameter.bindings.append((node_id, input_name))
        return parameters

    def _parse_placeholder(self, value):
        if not isinstance(value, str) or not value.startswith(PLACEHOLDER_PREFIX):
            return None
        token = value[len(PLACEHOLDER_PREFIX) :]
        annotation = str
        if "_" in token:
            type_candidate, remainder = token.split("_", 1)
            type_hint = PLACEHOLDER_TYPE_HINTS.get(type_candidate.upper())
            if type_hint:
                annotation = type_hint
                token = remainder
        param_name = self._normalize_name(token)
        return param_name, annotation, value

    def _normalize_name(self, raw: str):
        cleaned = [
            (char.lower() if char.isalnum() else "_")
            for char in raw.strip()
        ]
        normalized = "".join(cleaned).strip("_")
        return normalized or "param"

    def _derive_tool_name(self, stem: str):
        return self._normalize_name(stem)

    def _dedupe_tool_name(self, base_name: str):
        name = base_name or "workflow_tool"
        if name not in self._tool_names:
            self._tool_names.add(name)
            return name
        suffix = 2
        while f"{name}_{suffix}" in self._tool_names:
            suffix += 1
        deduped = f"{name}_{suffix}"
        self._tool_names.add(deduped)
        return deduped

    def _derive_description(self, stem: str):
        readable = stem.replace("_", " ").replace("-", " ").strip()
        readable = readable if readable else stem
        return f"Execute the '{readable}' ComfyUI workflow."

    def _determine_namespace(self, workflow_id: str) -> str:
        """Determine namespace based on workflow ID."""
        if workflow_id == "generate_song":
            return "audio"
        elif workflow_id == "generate_video":
            return "video"
        else:
            return "image"  # default fallback
    
    def _guess_output_preferences(self, workflow: Dict[str, Any]):
        for node in workflow.values():
            class_type = str(node.get("class_type", "")).lower()
            if "audio" in class_type:
                return AUDIO_OUTPUT_KEYS
            if "video" in class_type or "savevideo" in class_type or "videocombine" in class_type:
                return VIDEO_OUTPUT_KEYS
        return DEFAULT_OUTPUT_KEYS

    def _coerce_value(self, value: Any, annotation: type):
        """Coerce a value to the specified type with proper error handling."""
        try:
            if annotation is str:
                return str(value)
            if annotation is int:
                return int(value)
            if annotation is float:
                return float(value)
            if annotation is bool:
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.strip().lower() in {"1", "true", "yes", "y"}
                return bool(value)
            return value
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert {value!r} to {annotation.__name__}: {e}")

# Global ComfyUI client (fallback since context isn't available)
comfyui_url = os.getenv("COMFYUI_URL", "http://localhost:8188")
comfyui_client = ComfyUIClient(comfyui_url)
workflow_manager = WorkflowManager(WORKFLOW_DIR)
defaults_manager = DefaultsManager(comfyui_client)

# Define application context (for future use)
class AppContext:
    def __init__(self, comfyui_client: ComfyUIClient):
        self.comfyui_client = comfyui_client

# Lifespan management (placeholder for future context support)
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle"""
    logger.info("Starting MCP server lifecycle...")
    try:
        # Startup: Could add ComfyUI health check here in the future
        logger.info("ComfyUI client initialized globally")
        yield AppContext(comfyui_client=comfyui_client)
    finally:
        # Shutdown: Cleanup (if needed)
        logger.info("Shutting down MCP server")

# Initialize FastMCP with lifespan and port configuration
# Using port 9000 for consistency with previous version
# Enable stateless_http to avoid requiring session management
mcp = FastMCP(
    "ComfyUI_MCP_Server",
    lifespan=app_lifespan,
    port=9000,
    stateless_http=True
)


@mcp.tool()
def list_models() -> dict:
    """List all available checkpoint models in ComfyUI.
    
    Returns a list of model names that can be used with generation tools.
    This helps AI agents choose appropriate models for different use cases.
    """
    models = comfyui_client.available_models
    return {
        "models": models,
        "count": len(models),
        "default": "v1-5-pruned-emaonly.ckpt" if models else None
    }


@mcp.tool()
def get_defaults() -> dict:
    """Get current effective defaults for image, audio, and video generation.
    
    Returns merged defaults from all sources (runtime, config, env, hardcoded).
    Shows what values will be used when parameters are not explicitly provided.
    """
    return defaults_manager.get_all_defaults()


@mcp.tool()
def set_defaults(
    image: Optional[Dict[str, Any]] = None,
    audio: Optional[Dict[str, Any]] = None,
    video: Optional[Dict[str, Any]] = None,
    persist: bool = False
) -> dict:
    """Set runtime defaults for image, audio, and/or video generation.
    
    Args:
        image: Optional dict of default values for image generation (e.g., {"model": "sd_xl_base_1.0.safetensors", "width": 1024})
        audio: Optional dict of default values for audio generation (e.g., {"model": "ace_step_v1_3.5b.safetensors", "seconds": 30})
        video: Optional dict of default values for video generation (e.g., {"model": "wan2.2_vae.safetensors", "width": 1280, "duration": 5})
        persist: If True, write defaults to config file (~/.config/comfy-mcp/config.json). Otherwise, changes are ephemeral.
    
    Returns:
        Success status and any validation errors (e.g., invalid model names).
    """
    results = {}
    errors = []
    
    if image:
        result = defaults_manager.set_defaults("image", image, validate_models=True)
        if "error" in result or "errors" in result:
            errors.extend(result.get("errors", [result.get("error")]))
        else:
            results["image"] = result
            if persist:
                persist_result = defaults_manager.persist_defaults("image", image)
                if "error" in persist_result:
                    errors.append(f"Failed to persist image defaults: {persist_result['error']}")
    
    if audio:
        result = defaults_manager.set_defaults("audio", audio, validate_models=True)
        if "error" in result or "errors" in result:
            errors.extend(result.get("errors", [result.get("error")]))
        else:
            results["audio"] = result
            if persist:
                persist_result = defaults_manager.persist_defaults("audio", audio)
                if "error" in persist_result:
                    errors.append(f"Failed to persist audio defaults: {persist_result['error']}")
    
    if video:
        result = defaults_manager.set_defaults("video", video, validate_models=True)
        if "error" in result or "errors" in result:
            errors.extend(result.get("errors", [result.get("error")]))
        else:
            results["video"] = result
            if persist:
                persist_result = defaults_manager.persist_defaults("video", video)
                if "error" in persist_result:
                    errors.append(f"Failed to persist video defaults: {persist_result['error']}")
    
    if errors:
        return {"success": False, "errors": errors}
    
    return {"success": True, "updated": results}


@mcp.tool()
def list_workflows() -> dict:
    """List all available workflows in the workflow directory.
    
    Returns a catalog of workflows with their IDs, names, descriptions,
    available inputs, and optional metadata.
    """
    catalog = workflow_manager.get_workflow_catalog()
    return {
        "workflows": catalog,
        "count": len(catalog),
        "workflow_dir": str(workflow_manager.workflows_dir)
    }


@mcp.tool()
def run_workflow(
    workflow_id: str,
    overrides: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None
) -> dict:
    """Run a saved ComfyUI workflow with constrained parameter overrides.
    
    Args:
        workflow_id: The workflow ID (filename stem, e.g., "generate_image")
        overrides: Optional dict of parameter overrides (e.g., {"prompt": "a cat", "width": 1024})
        options: Optional dict of execution options (reserved for future use)
    
    Returns:
        Result with asset_url, workflow_id, and execution metadata.
    """
    if overrides is None:
        overrides = {}
    
    # Load workflow
    workflow = workflow_manager.load_workflow(workflow_id)
    if not workflow:
        return {"error": f"Workflow '{workflow_id}' not found"}
    
    try:
        # Apply overrides with constraints
        workflow = workflow_manager.apply_workflow_overrides(
            workflow, workflow_id, overrides, defaults_manager
        )
        
        # Determine output preferences
        output_preferences = workflow_manager._guess_output_preferences(workflow)
        
        # Execute workflow
        result = comfyui_client.run_custom_workflow(
            workflow,
            preferred_output_keys=output_preferences,
        )
        
        response_data = {
            "asset_url": result["asset_url"],
            "image_url": result["asset_url"],  # Backward compatibility
            "workflow_id": workflow_id,
            "prompt_id": result.get("prompt_id"),
        }
        
        # Include base64 image data if available
        if "image_base64" in result:
            response_data["image_base64"] = result["image_base64"]
            response_data["image_mime_type"] = result.get("image_mime_type", "image/png")
        
        return response_data
    except Exception as exc:
        logger.exception("Workflow '%s' failed", workflow_id)
        return {"error": str(exc)}


def _register_workflow_tool(definition: WorkflowToolDefinition):
    def _tool_impl(*args, **kwargs):
        # Coerce parameter types before signature binding
        # MCP/JSON-RPC may pass numbers as strings, so we need to convert them
        coerced_kwargs = {}
        param_dict = {p.name: p for p in definition.parameters.values()}
        
        for key, value in kwargs.items():
            if key in param_dict:
                param = param_dict[key]
                # Coerce to correct type if needed
                if value is not None:
                    try:
                        # Handle string representations of numbers
                        if param.annotation is int:
                            if isinstance(value, str) and value.strip().isdigit():
                                coerced_kwargs[key] = int(value)
                            elif isinstance(value, (int, float)):
                                coerced_kwargs[key] = int(value)
                            else:
                                coerced_kwargs[key] = value
                        elif param.annotation is float:
                            if isinstance(value, str):
                                coerced_kwargs[key] = float(value)
                            elif isinstance(value, (int, float)):
                                coerced_kwargs[key] = float(value)
                            else:
                                coerced_kwargs[key] = value
                        else:
                            coerced_kwargs[key] = value
                    except (ValueError, TypeError) as e:
                        # If coercion fails, use original value and let validation handle it
                        logger.warning(f"Failed to coerce {key}={value!r} to {param.annotation.__name__}: {e}")
                        coerced_kwargs[key] = value
                else:
                    coerced_kwargs[key] = None
            else:
                # Unknown parameter, pass through
                coerced_kwargs[key] = value
        
        bound = _tool_impl.__signature__.bind(*args, **coerced_kwargs)
        bound.apply_defaults()
        try:
            workflow = workflow_manager.render_workflow(definition, dict(bound.arguments), defaults_manager)
            result = comfyui_client.run_custom_workflow(
                workflow,
                preferred_output_keys=definition.output_preferences,
            )
            return {
                "asset_url": result["asset_url"],
                "image_url": result["asset_url"],  # Backward compatibility
                "workflow_id": definition.workflow_id,
                "tool": definition.tool_name,
            }
            
        except Exception as exc:
            logger.exception("Workflow '%s' failed", definition.workflow_id)
            return {"error": str(exc)}

    # Separate required and optional parameters to ensure correct ordering
    required_params = []
    optional_params = []
    annotations: Dict[str, Any] = {}
    
    for param in definition.parameters.values():
        # For numeric types, use Any to allow string coercion from JSON-RPC
        # FastMCP/Pydantic validation is strict, so we accept Any and validate/coerce ourselves
        if param.annotation in (int, float):
            # Use Any to bypass strict type checking, we'll coerce in the function
            annotation_type = Any
        else:
            annotation_type = param.annotation
        
        if param.required:
            parameter = inspect.Parameter(
                name=param.name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=annotation_type,
            )
            required_params.append(parameter)
        else:
            # Optional parameter with default value
            # For numeric types, use Any directly (not Optional[Any]) to allow string coercion
            if param.annotation in (int, float):
                final_annotation = Any
            else:
                final_annotation = Optional[annotation_type]
            parameter = inspect.Parameter(
                name=param.name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=final_annotation,
                default=None,
            )
            optional_params.append(parameter)
        annotations[param.name] = param.annotation
    
    # Combine: required parameters first, then optional
    parameters = required_params + optional_params
    annotations["return"] = dict
    _tool_impl.__signature__ = inspect.Signature(parameters, return_annotation=dict)
    _tool_impl.__annotations__ = annotations
    _tool_impl.__name__ = f"tool_{definition.tool_name}"
    _tool_impl.__doc__ = definition.description
    mcp.tool(name=definition.tool_name, description=definition.description)(_tool_impl)
    logger.info(
        "Registered MCP tool '%s' for workflow '%s'",
        definition.tool_name,
        definition.workflow_id,
    )


if workflow_manager.tool_definitions:
    for tool_definition in workflow_manager.tool_definitions:
        _register_workflow_tool(tool_definition)
else:
    logger.info(
        "No workflow placeholders found in %s; add %s markers to enable auto tools",
        WORKFLOW_DIR,
        PLACEHOLDER_PREFIX,
    )

if __name__ == "__main__":
    import sys
    # Check if running as MCP command (stdio) or standalone (streamable-http)
    # When run as command by MCP client (like Cursor), use stdio transport
    # When run standalone, use streamable-http for HTTP access
    if len(sys.argv) > 1 and sys.argv[1] == "--stdio":
        logger.info("Starting MCP server with stdio transport (for MCP clients)")
        mcp.run(transport="stdio")
    else:
        logger.info("Starting MCP server with streamable-http transport on http://127.0.0.1:9000/mcp")
        mcp.run(transport="streamable-http")
