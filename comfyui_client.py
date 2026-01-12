import requests
import json
import time
import logging
from typing import Any, Dict, Sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ComfyUIClient")

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
                # Try both the specific prompt_id endpoint and the full history endpoint
                response = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=10)
                # If that doesn't work, we can also try: f"{self.base_url}/history"
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
                    # Workflow might still be running, wait and retry
                    if attempt < max_attempts - 1:
                        time.sleep(1)
                        continue
                    else:
                        # Last attempt - check if there's any history at all
                        logger.warning("Prompt ID not found in history. Available IDs: %s", list(history.keys())[:10])
                        time.sleep(1)
                        continue
                
                prompt_data = history[prompt_id]
                if not isinstance(prompt_data, dict):
                    logger.warning("Prompt data is not a dict on attempt %s", attempt + 1)
                    time.sleep(1)
                    continue
                
                # Check for workflow errors
                if "error" in prompt_data:
                    error_info = prompt_data["error"]
                    raise Exception(f"Workflow failed with error: {json.dumps(error_info, indent=2)}")
                
                # Check if workflow status indicates failure
                status = prompt_data.get("status", {})
                if isinstance(status, dict) and status.get("completed") == False:
                    error_msg = status.get("messages", ["Workflow failed"])
                    raise Exception(f"Workflow failed: {error_msg}")
                
                # Get outputs
                if "outputs" not in prompt_data:
                    # Check status to see if workflow completed
                    status = prompt_data.get("status", [])
                    if isinstance(status, list) and len(status) > 0:
                        last_status = status[-1]
                        if isinstance(last_status, list) and len(last_status) > 0:
                            status_type = last_status[0]
                            if status_type == "execution_success":
                                # Workflow completed successfully but outputs not yet available
                                # Wait a bit longer for outputs to be written, especially for cached executions
                                logger.info("Workflow execution succeeded, waiting for outputs to be available...")
                                time.sleep(3)  # Give ComfyUI time to write outputs (longer for cached)
                                # Try fetching full history to see if outputs appear there
                                try:
                                    full_history_response = requests.get(f"{self.base_url}/history", timeout=10)
                                    if full_history_response.status_code == 200:
                                        full_history = full_history_response.json()
                                        if prompt_id in full_history:
                                            full_prompt_data = full_history[prompt_id]
                                            if "outputs" in full_prompt_data and full_prompt_data["outputs"]:
                                                logger.info("Found outputs in full history endpoint")
                                                return full_prompt_data["outputs"]
                                except Exception as e:
                                    logger.debug("Could not fetch full history: %s", e)
                                continue
                    logger.warning("Prompt data missing outputs on attempt %s. Full data: %s", attempt + 1, json.dumps(prompt_data, indent=2))
                    time.sleep(1)
                    continue
                
                outputs = prompt_data["outputs"]
                if not outputs or not isinstance(outputs, dict):
                    # Check if workflow actually succeeded
                    status = prompt_data.get("status", [])
                    if isinstance(status, list):
                        status_messages = [s[0] if isinstance(s, list) else str(s) for s in status]
                        if "execution_success" in status_messages:
                            # Workflow succeeded but no outputs - might need to check queue or wait longer
                            logger.warning("Workflow succeeded but outputs empty. Status: %s. Waiting longer...", status_messages)
                            time.sleep(2)
                            continue
                        raise Exception(f"Workflow completed but produced no outputs. Status: {status_messages}")
                    logger.warning("Outputs is empty or not a dict. Prompt data: %s", json.dumps(prompt_data, indent=2))
                    raise Exception("Workflow completed but produced no outputs. Check ComfyUI logs for errors.")
                
                logger.info("Workflow completed. Output nodes: %s", list(outputs.keys()))
                logger.debug("Full workflow outputs: %s", json.dumps(outputs, indent=2))
                logger.debug("Full prompt data: %s", json.dumps(prompt_data, indent=2))
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
        # Log available outputs for debugging
        logger.debug("Available output keys in workflow: %s", list(outputs.keys()))
        for node_id, node_output in outputs.items():
            if not isinstance(node_output, dict):
                logger.debug("Node %s output is not a dict: %s", node_id, type(node_output))
                continue
            logger.debug("Node %s has keys: %s", node_id, list(node_output.keys()))
            for key in preferred_output_keys:
                assets = node_output.get(key)
                if assets and isinstance(assets, list) and len(assets) > 0:
                    asset = assets[0]
                    if not isinstance(asset, dict):
                        logger.debug("Asset in node %s, key %s is not a dict", node_id, key)
                        continue
                    filename = asset.get("filename")
                    if not filename:
                        logger.debug("Asset in node %s, key %s missing filename", node_id, key)
                        continue
                    subfolder = asset.get("subfolder", "")
                    output_type = asset.get("type", "output")
                    logger.info("Found asset: filename=%s, subfolder=%s, type=%s", filename, subfolder, output_type)
                    return f"{self.base_url}/view?filename={filename}&subfolder={subfolder}&type={output_type}"
        
        # Enhanced error message with actual output structure
        logger.error("No outputs matched preferred keys: %s", preferred_output_keys)
        logger.error("Actual outputs structure: %s", json.dumps(outputs, indent=2))
        raise Exception(
            f"No outputs matched preferred keys: {preferred_output_keys}. "
            f"Available outputs: {json.dumps({k: list(v.keys()) if isinstance(v, dict) else type(v).__name__ for k, v in outputs.items()}, indent=2)}"
        )