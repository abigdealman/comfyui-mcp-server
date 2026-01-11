import requests
import json
import time
import logging
from typing import Any, Dict, Sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ComfyUIClient")

DEFAULT_MAPPING = {
    "prompt": ("6", "text"),
    "width": ("5", "width"),
    "height": ("5", "height"),
    "model": ("4", "ckpt_name")
}

class ComfyUIClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.available_models = self._get_available_models()

    def _get_available_models(self):
        """Fetch list of available checkpoint models from ComfyUI"""
        try:
            response = requests.get(f"{self.base_url}/object_info/CheckpointLoaderSimple", timeout=10)
            if response.status_code != 200:
                logger.warning("Failed to fetch model list; using default handling")
                return []
            data = response.json()
            # Safe dictionary access with proper error handling
            try:
                checkpoint_info = data.get("CheckpointLoaderSimple", {})
                if not isinstance(checkpoint_info, dict):
                    logger.warning("Unexpected CheckpointLoaderSimple structure")
                    return []
                input_info = checkpoint_info.get("input", {})
                if not isinstance(input_info, dict):
                    logger.warning("Unexpected input structure")
                    return []
                required_info = input_info.get("required", {})
                if not isinstance(required_info, dict):
                    logger.warning("Unexpected required structure")
                    return []
                ckpt_name_info = required_info.get("ckpt_name", [])
                if not isinstance(ckpt_name_info, list) or len(ckpt_name_info) == 0:
                    logger.warning("No checkpoint models found in API response")
                    return []
                models = ckpt_name_info[0] if isinstance(ckpt_name_info[0], list) else ckpt_name_info
                logger.info(f"Available models: {models}")
                return models
            except (KeyError, IndexError, TypeError) as e:
                logger.warning(f"Unexpected API response structure: {e}")
                return []
        except requests.RequestException as e:
            logger.warning(f"Error fetching models: {e}")
            return []

    def generate_image(self, prompt, width, height, workflow_id="basic_api_test", model=None):
        try:
            # Prevent path traversal attacks by validating workflow_id
            from pathlib import Path
            workflows_dir = Path("workflows").resolve()
            workflow_path = (workflows_dir / f"{workflow_id}.json").resolve()
            # Ensure the resolved path is within workflows_dir
            try:
                workflow_path.relative_to(workflows_dir)
            except ValueError:
                raise ValueError(f"Invalid workflow_id: {workflow_id} (path traversal detected)")
            
            if not workflow_path.exists():
                raise FileNotFoundError(f"Workflow file '{workflow_path}' not found")
            
            with open(workflow_path, "r", encoding="utf-8") as f:
                workflow = json.load(f)

            params = {"prompt": prompt, "width": width, "height": height}
            if model:
                # Validate or correct model name
                if model.endswith("'"):  # Strip accidental quote
                    model = model.rstrip("'")
                    logger.info(f"Corrected model name: {model}")
                if self.available_models and model not in self.available_models:
                    raise Exception(f"Model '{model}' not in available models: {self.available_models}")
                params["model"] = model

            for param_key, value in params.items():
                if param_key in DEFAULT_MAPPING:
                    node_id, input_key = DEFAULT_MAPPING[param_key]
                    if node_id not in workflow:
                        raise Exception(f"Node {node_id} not found in workflow {workflow_id}")
                    workflow[node_id]["inputs"][input_key] = value

            result = self.run_custom_workflow(
                workflow,
                preferred_output_keys=("images", "image", "gifs", "gif")
            )
            logger.info(f"Generated image URL: {result['asset_url']}")
            return result["asset_url"]

        except FileNotFoundError:
            raise Exception(f"Workflow file '{workflow_file}' not found")
        except KeyError as e:
            raise Exception(f"Workflow error - invalid node or input: {e}")
        except requests.RequestException as e:
            raise Exception(f"ComfyUI API error: {e}")

    def run_custom_workflow(self, workflow: Dict[str, Any], preferred_output_keys: Sequence[str] | None = None, max_attempts: int = 30):
        if preferred_output_keys is None:
            preferred_output_keys = ("images", "image", "gifs", "gif", "audio", "audios", "files")

        prompt_id = self._queue_workflow(workflow)
        outputs = self._wait_for_prompt(prompt_id, max_attempts=max_attempts)
        asset_url = self._extract_first_asset_url(outputs, preferred_output_keys)
        return {
            "asset_url": asset_url,
            "prompt_id": prompt_id,
            "raw_outputs": outputs
        }

    def _queue_workflow(self, workflow: Dict[str, Any]):
        logger.info("Submitting workflow to ComfyUI...")
        response = requests.post(f"{self.base_url}/prompt", json={"prompt": workflow}, timeout=30)
        if response.status_code != 200:
            raise Exception(f"Failed to queue workflow: {response.status_code} - {response.text}")
        try:
            response_data = response.json()
            prompt_id = response_data.get("prompt_id")
            if not prompt_id:
                raise Exception("Response missing prompt_id")
        except (KeyError, ValueError) as e:
            raise Exception(f"Invalid response format from ComfyUI: {e}")
        logger.info(f"Queued workflow with prompt_id: {prompt_id}")
        return prompt_id

    def _wait_for_prompt(self, prompt_id: str, max_attempts: int = 30):
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=10)
                if response.status_code != 200:
                    logger.warning("History endpoint returned %s on attempt %s", response.status_code, attempt + 1)
                    time.sleep(1)
                    continue
                
                history = response.json()
                if not isinstance(history, dict):
                    logger.warning("Invalid history response format on attempt %s", attempt + 1)
                    time.sleep(1)
                    continue
                
                if prompt_id not in history:
                    time.sleep(1)
                    continue
                
                prompt_data = history[prompt_id]
                if not isinstance(prompt_data, dict) or "outputs" not in prompt_data:
                    logger.warning("Prompt data missing outputs on attempt %s", attempt + 1)
                    time.sleep(1)
                    continue
                
                outputs = prompt_data["outputs"]
                logger.info("Workflow outputs: %s", json.dumps(outputs, indent=2))
                return outputs
            except requests.RequestException as e:
                logger.warning("Request error on attempt %s: %s", attempt + 1, e)
                time.sleep(1)
                continue
            except (ValueError, KeyError) as e:
                logger.warning("JSON parsing error on attempt %s: %s", attempt + 1, e)
                time.sleep(1)
                continue
        
        raise Exception(f"Workflow {prompt_id} didn't complete within {max_attempts} seconds")

    def _extract_first_asset_url(self, outputs: Dict[str, Any], preferred_output_keys: Sequence[str]):
        for node_output in outputs.values():
            if not isinstance(node_output, dict):
                continue
            for key in preferred_output_keys:
                assets = node_output.get(key)
                if assets and isinstance(assets, list) and len(assets) > 0:
                    asset = assets[0]
                    if not isinstance(asset, dict):
                        continue
                    filename = asset.get("filename")
                    if not filename:
                        continue
                    subfolder = asset.get("subfolder", "")
                    output_type = asset.get("type", "output")
                    return f"{self.base_url}/view?filename={filename}&subfolder={subfolder}&type={output_type}"
        raise Exception(f"No outputs matched preferred keys: {preferred_output_keys}")
