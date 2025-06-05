"""
RAVEN Visual Reasoning Evaluation Pipeline

This module provides a complete pipeline for evaluating visual reasoning models on the 
RAVEN dataset using Azure OpenAI's GPT-4V. The RAVEN dataset contains visual analogy 
problems presented as 3x3 matrices with one missing panel that must be completed by 
selecting from 8 multiple choice options.

Dataset Structure:
- Panels: 8 PIL images representing the 3x3 matrix (missing bottom-right)
- Choices: 8 PIL images representing answer options (A-H)
- Target: Integer (0-7) indicating correct choice index
- Splits: Train (6000), Validation (2000), Test (2000)

Usage:
    from load_raven import RAVENRunner
    
    evaluator = RAVENRunner(
        model_name="azure/gpt-4.1"
    )
    
    results = evaluator.evaluate_dataset("validation", max_samples=100)
    print(f"Accuracy: {results['accuracy']:.3f}")
"""

import base64
import json
import time
import os
from io import BytesIO
from typing import List, Dict, Tuple, Optional, Union
import logging
from datetime import datetime

from datasets import load_dataset
from PIL import Image

try:
    import litellm
except ImportError:
    raise ImportError("Please install litellm: pip install litellm")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure LLM logging
log_directory = os.getenv("LOG_DIR", "logs")
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(log_directory, f"llm_calls_{datetime.now().strftime('%Y%m%d')}.log")

# Set up LLM logger
llm_logger = logging.getLogger("llm_logger")
llm_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
llm_logger.addHandler(file_handler)

# Simple cache configuration
cache_file = "llm_cache.json"


class RAVENRunner:
    """
    Evaluator for RAVEN visual reasoning dataset using Azure OpenAI GPT-4V.
    
    This class handles:
    - Loading RAVEN dataset splits
    - Converting PIL images to base64 for API calls
    - Constructing visual prompts with matrix layout
    - Querying Azure OpenAI with retry logic
    - Parsing responses and calculating accuracy metrics
    """
    
    def __init__(
        self, 
        model_name: str = "azure/gpt-4.1",
        max_retries: int = 3,
        reasoning_effort: str = "high"
    ):
        """
        Initialize the RAVEN evaluator.
        
        Args:
            model_name: Model to use (default: "azure/gpt-4.1", also supports Claude models)
            max_retries: Maximum number of API call retries
            reasoning_effort: Reasoning effort level for supported models
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        
        # Load datasets
        logger.info("Loading RAVEN dataset...")
        self.datasets = {
            'train': load_dataset("HuggingFaceM4/RAVEN", "center_single", split="train"),
            'validation': load_dataset("HuggingFaceM4/RAVEN", "center_single", split="validation"),
            'test': load_dataset("HuggingFaceM4/RAVEN", "center_single", split="test")
        }
        logger.info(f"Loaded datasets: {[(k, len(v)) for k, v in self.datasets.items()]}")
    
    @staticmethod
    def encode_image_to_base64(pil_image: Image.Image) -> str:
        """
        Convert PIL image to base64 string for API calls.
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            Base64 encoded string of the image
        """
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    @staticmethod
    def create_composite_matrix_image(panels: List[Image.Image]) -> Image.Image:
        """
        Create a composite image showing the 8 panels arranged in a 3x3 grid layout
        with borders around each panel, matching the reference style.
        
        Args:
            panels: List of 8 PIL images representing the matrix panels
            
        Returns:
            PIL Image showing the 3x3 matrix with missing bottom-right panel
        """
        # Assume all panels are the same size
        panel_width, panel_height = panels[0].size
        
        # Add border width and spacing
        border_width = 2
        spacing = 4
        
        # Calculate dimensions with borders and spacing
        cell_width = panel_width + 2 * border_width
        cell_height = panel_height + 2 * border_width
        
        # Add margin for the "Problem Matrix" text on the left side
        left_margin = 60  # Margin for the vertical text - increased from 40 to 60
        
        composite_width = cell_width * 3 + spacing * 2 + left_margin
        composite_height = cell_height * 3 + spacing * 2
        
        # Create composite image with white background
        composite = Image.new('RGB', (composite_width, composite_height), 'white')
        
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(composite)
        
        # Arrange panels in 3x3 grid (missing bottom-right)
        positions = [
            (0, 0),      # Panel 1: top-left
            (1, 0),      # Panel 2: top-center  
            (2, 0),      # Panel 3: top-right
            (0, 1),      # Panel 4: middle-left
            (1, 1),      # Panel 5: middle-center
            (2, 1),      # Panel 6: middle-right
            (0, 2),      # Panel 7: bottom-left
            (1, 2),      # Panel 8: bottom-center
            # (2, 2) is missing - bottom-right
        ]
        
        for i, (col, row) in enumerate(positions):
            # Calculate position with spacing, accounting for left margin
            x = left_margin + col * (cell_width + spacing)
            y = row * (cell_height + spacing)
            
            # Draw border rectangle
            draw.rectangle([x, y, x + cell_width, y + cell_height], 
                          outline='black', fill='white', width=border_width)
            
            # Paste the panel inside the border
            panel_x = x + border_width
            panel_y = y + border_width
            composite.paste(panels[i], (panel_x, panel_y))
        
        # Add question mark for missing panel
        missing_col, missing_row = 2, 2
        missing_x = left_margin + missing_col * (cell_width + spacing)
        missing_y = missing_row * (cell_height + spacing)
        
        # Draw border for missing panel
        draw.rectangle([missing_x, missing_y, missing_x + cell_width, missing_y + cell_height], 
                      outline='black', fill='white', width=border_width)
        
        # Add question mark in the center
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 
                                    min(panel_width, panel_height) // 3)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        question_text = "?"
        if font:
            bbox = draw.textbbox((0, 0), question_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width, text_height = 20, 20
        
        text_x = missing_x + (cell_width - text_width) // 2
        text_y = missing_y + (cell_height - text_height) // 2
        
        draw.text((text_x, text_y), question_text, fill='black', font=font)
        
        # Add "Problem Matrix" text on the left side
        try:
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except:
            try:
                label_font = ImageFont.load_default()
            except:
                label_font = None
        
        if label_font:
            problem_matrix_text = "Problem Matrix"
            text_width, text_height = draw.textbbox((0, 0), problem_matrix_text, font=label_font)[2:4]
            
            # Calculate the position for the vertical text
            # Center it vertically in the matrix area
            matrix_height = 3 * cell_height + 2 * spacing
            text_x = 25  # Fixed distance from the left edge - increased from 15 to 25
            text_y = (composite_height - text_width) // 2  # Center vertically
            
            # Draw the rotated text
            # We need to create a temporary image, draw text, then rotate and paste
            text_img = Image.new('RGBA', (text_width, text_height), (255, 255, 255, 0))
            text_draw = ImageDraw.Draw(text_img)
            text_draw.text((0, 0), problem_matrix_text, fill='black', font=label_font)
            
            # Rotate the text image 90 degrees counter-clockwise
            rotated_text = text_img.rotate(90, expand=True)
            
            # Paste the rotated text onto the composite image
            composite.paste(rotated_text, (text_x, text_y), rotated_text)
        
        return composite

    @staticmethod
    def create_choices_grid(choices: List[Image.Image]) -> Image.Image:
        """
        Create a grid showing the 8 answer choices with index labels below each choice,
        matching the reference style with 2 rows of 4 choices each.
        
        Args:
            choices: List of 8 PIL images representing answer choices
            
        Returns:
            PIL Image showing choices arranged in 2x4 grid with labels below
        """
        choice_width, choice_height = choices[0].size
        
        # Grid configuration: 2 rows, 4 columns
        grid_cols, grid_rows = 4, 2
        
        # Add border width and spacing
        border_width = 2
        spacing = 4
        label_height = 30  # Space for number labels below choices
        
        # Calculate cell dimensions
        cell_width = choice_width + 2 * border_width
        cell_height = choice_height + 2 * border_width + label_height
        
        # Calculate total dimensions
        composite_width = cell_width * grid_cols + spacing * (grid_cols - 1)
        composite_height = cell_height * grid_rows + spacing * (grid_rows - 1)
        
        composite = Image.new('RGB', (composite_width, composite_height), 'white')
        
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(composite)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        for i, choice in enumerate(choices):
            col = i % grid_cols
            row = i // grid_cols
            
            # Calculate position with spacing
            x = col * (cell_width + spacing)
            y = row * (cell_height + spacing)
            
            # Draw border rectangle around choice
            choice_border_height = choice_height + 2 * border_width
            draw.rectangle([x, y, x + cell_width, y + choice_border_height], 
                          outline='black', fill='white', width=border_width)
            
            # Paste the choice image inside the border
            choice_x = x + border_width
            choice_y = y + border_width
            composite.paste(choice, (choice_x, choice_y))
            
            # Add index label below the choice
            label = str(i + 1)  # 1-indexed for display
            if font:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width, text_height = 10, 12
            
            # Center the label below the choice
            label_x = x + (cell_width - text_width) // 2
            label_y = y + choice_border_height + (label_height - text_height) // 2
            
            draw.text((label_x, label_y), label, fill='black', font=font)
        
        return composite

    @staticmethod
    def create_combined_image(matrix_composite: Image.Image, choices_composite: Image.Image) -> Image.Image:
        """
        Combine matrix composite and choices composite into a single image matching
        the reference layout with section labels and proper spacing.
        
        Args:
            matrix_composite: PIL Image showing the 3x3 matrix
            choices_composite: PIL Image showing the 8 choices
            
        Returns:
            PIL Image with labeled sections: Problem Matrix and Answer Set
        """
        from PIL import ImageDraw, ImageFont
        
        # Calculate dimensions
        matrix_width, matrix_height = matrix_composite.size
        choices_width, choices_height = choices_composite.size
        
        # Layout parameters
        margin = 30  # Margin around the entire image
        section_spacing = 40  # Space between Problem Matrix and Answer Set
        label_height = 40  # Height for section labels (increased)
        label_margin = 15  # Space between label and content (increased)
        
        # Add margin for the left side labels
        left_margin = 120  # Increased from 90 to 120 for more space for vertical text
        
        # Calculate total dimensions
        combined_width = max(matrix_width, choices_width) + 2 * margin + left_margin
        combined_height = (margin + label_height + label_margin + matrix_height + 
                          section_spacing + label_height + label_margin + 
                          choices_height + margin)
        
        # Create combined image with white background
        combined = Image.new('RGB', (combined_width, combined_height), 'white')
        draw = ImageDraw.Draw(combined)
        
        # Set up fonts - using bold font for both
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except:
            try:
                title_font = ImageFont.load_default()
                label_font = ImageFont.load_default()
            except:
                title_font = None
                label_font = None
        
        # Current y position
        current_y = margin
        
        # Add "(a)" label at top left
        if title_font:
            draw.text((margin, current_y), "(a)", fill='black', font=title_font)
        current_y += label_height + label_margin
        
        # We don't need to add "Problem Matrix" label here since it's now part of the matrix_composite
        
        # Center and paste the matrix
        matrix_x = margin + (combined_width - matrix_width - 2*margin) // 2
        combined.paste(matrix_composite, (matrix_x, current_y))
        current_y += matrix_height + section_spacing
        
        # Add "Answer Set" label - positioned on the left side vertically
        if label_font:
            answer_set_label = "Answer Set"
            # Create a temporary image for the rotated text
            bbox = draw.textbbox((0, 0), answer_set_label, font=label_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Create temporary image for text with transparent background
            text_img = Image.new('RGBA', (text_width + 10, text_height + 10), (255, 255, 255, 0))  # Added padding
            text_draw = ImageDraw.Draw(text_img)
            text_draw.text((5, 5), answer_set_label, fill='black', font=label_font)  # Add some padding inside the text image
            
            # Rotate the text image 90 degrees counter-clockwise
            rotated_text = text_img.rotate(90, expand=True)
            
            # Position the rotated text on the left side, centered vertically with choices grid
            # Ensure text is more centered and not cut off
            label_x = 45  # Increased from 35 to 45 for better spacing
            
            # Calculate better vertical centering based on the actual choices area
            # Make sure the text is fully visible within the choices section
            choices_center_y = current_y + choices_height / 2
            rotated_text_height = rotated_text.size[1]
            label_y = choices_center_y - rotated_text_height / 2
            
            # Ensure the text doesn't go outside the image boundaries
            if label_y < current_y:
                label_y = current_y
            if label_y + rotated_text_height > current_y + choices_height:
                label_y = current_y + choices_height - rotated_text_height
                
            combined.paste(rotated_text, (label_x, int(label_y)), rotated_text)
        
        # Center and paste the choices
        choices_x = margin + (combined_width - choices_width - 2*margin) // 2
        combined.paste(choices_composite, (choices_x, current_y))
        
        return combined

    def create_raven_prompt(self, panels: List[Image.Image], choices: List[Image.Image]) -> Tuple[str, Image.Image]:
        """
        Create visual prompt for RAVEN task with a single combined image.
        
        The prompt presents:
        1. A single combined image showing the 3x3 matrix at the top and 8 choices below
        2. Instructions to select the best completion by index
        
        Args:
            panels: List of 8 PIL images representing the matrix panels
            choices: List of 8 PIL images representing answer choices
            
        Returns:
            Tuple of (prompt_text, [combined_image])
        """
        # Create composite images
        matrix_composite = self.create_composite_matrix_image(panels)
        choices_composite = self.create_choices_grid(choices)
        
        # Combine both images into a single image
        combined_image = self.create_combined_image(matrix_composite, choices_composite)
        
        prompt = """You are shown a 3x3 matrix of images with the bottom-right panel missing (marked with "?"). Your task is to identify which of the 8 numbered choices (1-8) best completes the pattern.

Select the choice (1-8) that best completes the pattern. Respond with only the number (1, 2, 3, 4, 5, 6, 7, or 8)."""
        
        return prompt, combined_image
    
    def query_azure_openai(self, prompt: str, image: Image.Image) -> str:
        """
        Send visual prompt to LLM with retry logic using LiteLLM.
        
        Args:
            prompt: Text prompt describing the task
            images: List of PIL images (panels + choices)
            
        Returns:
            Model response text
            
        Raises:
            Exception: If all retry attempts fail
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.encode_image_to_base64(image)}"}}
                ]
            }
        ]
        
        # Log the prompt
        llm_logger.info(f"PROMPT: {prompt[:100]}...")
        llm_logger.info(f"MODEL: {self.model_name}, REASONING: {self.reasoning_effort}")
        
        # Create cache key from prompt and model
        cache_key = f"{self.model_name}:{str(prompt)}"
        
        # Check cache
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
                if cache_key in cache:
                    llm_logger.info(f"Cache hit for prompt: {str(prompt)[:50]}...")
                    return cache[cache_key]
            except:
                llm_logger.warning(f"Failed to load cache, starting with empty cache")
        
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                # Prepare completion parameters
                completion_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": 10,  # Increased from 10 to allow proper responses
                }
                
                # Azure OpenAI support
                if self.model_name.startswith("azure/"):
                    completion_params["api_key"] = os.getenv("AZURE_API_KEY")
                    completion_params["api_base"] = os.getenv("AZURE_API_BASE", "https://dalle-declare.openai.azure.com/")
                    completion_params["api_version"] = os.getenv("AZURE_API_VERSION", "2025-01-01-preview")
                
                # Set up Vertex AI credentials if using vertex_ai models
                if self.model_name.startswith("vertex_ai/"):
                    completion_params["vertex_project"] = os.getenv("ANTHROPIC_PROJECT_ID", "your-project-id")
                    completion_params["vertex_location"] = os.getenv("ANTHROPIC_REGION", "us-east5")
                
                # Add thinking configuration for Claude models
                if "claude" in self.model_name.lower() and "3-7" in self.model_name:
                    thinking_budgets = {
                        "low": 1024,
                        "medium": 4096,
                        "high": 16000
                    }
                    completion_params["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": thinking_budgets.get(self.reasoning_effort, 16000)
                    }
                    completion_params["temperature"] = 1.0
                elif self.reasoning_effort and self.reasoning_effort.lower() in ["low", "medium", "high"]:
                    if not self.model_name.startswith("azure/"):
                        completion_params["reasoning_effort"] = self.reasoning_effort.lower()
                    completion_params["temperature"] = 1.0
                else:
                    completion_params["temperature"] = 0.1
                
                # Call the LLM
                response = litellm.completion(**completion_params)
                
                # Check if response is valid
                if not response or not response.choices or len(response.choices) == 0:
                    raise Exception("Empty response from API")
                
                response_content = response.choices[0].message.content
                response_text = response_content.strip() if response_content else ""
                
                # Log the response
                llm_logger.info(f"RESPONSE: {response_text}")
                
                # Check if response is empty
                if not response_text:
                    logger.warning("Received empty response from API")
                    # Don't cache empty responses, let it retry
                    if attempt < self.max_retries - 1:
                        continue
                
                # Update cache
                cache[cache_key] = response_text
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(cache, f, indent=2)
                    llm_logger.info(f"Added to cache")
                except Exception as e:
                    llm_logger.error(f"Failed to save cache: {e}")
                
                return response_text
                
            except Exception as e:
                logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
    
    @staticmethod
    def parse_response(response_text: str) -> int:
        """
        Extract choice number from response and convert to zero-indexed.
        
        Args:
            response_text: Raw response from the model
            
        Returns:
            Choice index (0-7) or -1 if invalid response
        """
        if not response_text:
            logger.warning("Empty response received")
            return -1
            
        response_text = response_text.strip()
        
        # Handle single digit responses
        if len(response_text) == 1 and response_text.isdigit():
            choice_num = int(response_text)
            if 1 <= choice_num <= 8:
                return choice_num - 1  # Convert to zero-indexed
        
        # Look for patterns like "The answer is 5" or "Choice: 3"
        import re
        patterns = [
            r'\b([1-8])\b',  # Any digit 1-8 surrounded by word boundaries
            r'answer\s+is\s+([1-8])',  # "answer is X"
            r'choice\s+([1-8])',  # "choice X"
            r'option\s+([1-8])',  # "option X"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_text.lower())
            if matches:
                choice_num = int(matches[0])
                if 1 <= choice_num <= 8:
                    return choice_num - 1  # Convert to zero-indexed
        
        # Look for valid digit in response (fallback)
        for char in response_text:
            if char.isdigit():
                choice_num = int(char)
                if 1 <= choice_num <= 8:
                    return choice_num - 1  # Convert to zero-indexed
        
        logger.warning(f"Invalid response format: '{response_text}'")
        return -1  # Invalid response
    
    @staticmethod
    def evaluate_predictions(predictions: List[int], targets: List[int]) -> Dict[str, Union[float, int]]:
        """
        Calculate accuracy metrics for predictions.
        
        Args:
            predictions: List of predicted choice indices
            targets: List of ground truth choice indices
            
        Returns:
            Dictionary with accuracy metrics
        """
        correct = sum(1 for pred, target in zip(predictions, targets) if pred == target)
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0
        invalid_responses = sum(1 for pred in predictions if pred == -1)
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'invalid_responses': invalid_responses,
            'valid_accuracy': correct / (total - invalid_responses) if (total - invalid_responses) > 0 else 0
        }
    
    def evaluate_dataset(
        self, 
        split: str = "validation", 
        max_samples: Optional[int] = None,
        save_results: bool = True,
        results_filename: Optional[str] = None
    ) -> Dict[str, Union[float, int]]:
        """
        Run evaluation on specified dataset split.
        
        Args:
            split: Dataset split to evaluate ("train", "validation", "test")
            max_samples: Maximum number of samples to evaluate (None for all)
            save_results: Whether to save detailed results to file
            results_filename: Custom filename for results (auto-generated if None)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if split not in self.datasets:
            raise ValueError(f"Invalid split: {split}. Available: {list(self.datasets.keys())}")
        
        dataset = self.datasets[split]
        samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))
        
        logger.info(f"Evaluating {len(samples)} samples from {split} split...")
        
        predictions = []
        targets = []
        detailed_results = []
        
        for i, example in enumerate(samples):
            try:
                # Create prompt and query API
                prompt, image = self.create_raven_prompt(example['panels'], example['choices'])
                response = self.query_azure_openai(prompt, image)
                
                # Parse response
                predicted_idx = self.parse_response(response)
                target_idx = example['target']
                
                predictions.append(predicted_idx)
                targets.append(target_idx)
                
                # Store detailed results
                detailed_results.append({
                    'sample_id': example['id'],
                    'predicted': predicted_idx,
                    'target': target_idx,
                    'correct': predicted_idx == target_idx,
                    'response_text': response,
                    'split': split
                })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(samples)} samples")
                    
            except Exception as e:
                logger.error(f"Error processing sample {i} (ID: {example.get('id', 'unknown')}): {e}")
                predictions.append(-1)  # Mark as invalid
                targets.append(example['target'])
                detailed_results.append({
                    'sample_id': example.get('id', i),
                    'predicted': -1,
                    'target': example['target'],
                    'correct': False,
                    'response_text': f"ERROR: {str(e)}",
                    'split': split
                })
        
        # Calculate metrics
        results = self.evaluate_predictions(predictions, targets)
        
        # Save detailed results if requested
        if save_results:
            if results_filename is None:
                results_filename = f"raven_results_{split}_{len(samples)}samples.json"
            
            with open(results_filename, 'w') as f:
                json.dump({
                    'metadata': {
                        'split': split,
                        'total_samples': len(samples),
                        'model': self.model_name,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    },
                    'metrics': results,
                    'detailed_results': detailed_results
                }, f, indent=2)
            
            logger.info(f"Detailed results saved to {results_filename}")
        
        return results
    
    def batch_evaluate(
        self, 
        split: str = "validation", 
        batch_size: int = 5,
        max_samples: Optional[int] = None
    ) -> Dict[str, Union[float, int]]:
        """
        Evaluate dataset in batches to respect API rate limits.
        
        Args:
            split: Dataset split to evaluate
            batch_size: Number of samples to process before rate limiting pause
            max_samples: Maximum total samples to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        if split not in self.datasets:
            raise ValueError(f"Invalid split: {split}. Available: {list(self.datasets.keys())}")
        
        dataset = self.datasets[split]
        samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))
        
        logger.info(f"Batch evaluating {len(samples)} samples (batch_size={batch_size})...")
        
        predictions = []
        targets = []
        
        for batch_start in range(0, len(samples), batch_size):
            batch_end = min(batch_start + batch_size, len(samples))
            batch_samples = samples.select(range(batch_start, batch_end))
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}: samples {batch_start}-{batch_end-1}")
            
            for example in batch_samples:
                try:
                    prompt, image = self.create_raven_prompt(example['panels'], example['choices'])
                    response = self.query_azure_openai(prompt, image)
                    predicted_idx = self.parse_response(response)
                    
                    predictions.append(predicted_idx)
                    targets.append(example['target'])
                    
                except Exception as e:
                    logger.error(f"Error in batch processing: {e}")
                    predictions.append(-1)
                    targets.append(example['target'])
            
            # Rate limiting pause between batches
            if batch_end < len(samples):
                logger.info("Pausing between batches...")
                time.sleep(2)
        
        return self.evaluate_predictions(predictions, targets)


def main():
    """
    Example usage of the RAVEN evaluator.
    
    Set your environment variables for the model you want to use:
    - For Azure: AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION
    - For Claude: ANTHROPIC_API_KEY, ANTHROPIC_PROJECT_ID, ANTHROPIC_REGION
    """
    # Initialize evaluator with default gpt-4.1 model
    evaluator = RAVENRunner(
        model_name="azure/gpt-4.1",  # or "vertex_ai/claude-3-7-sonnet@20250219"
        reasoning_effort="high"
    )
    
    # Run small validation test
    logger.info("Running validation test...")
    results = evaluator.evaluate_dataset("validation", max_samples=10)
    
    logger.info("Results:")
    logger.info(f"Accuracy: {results['accuracy']:.3f}")
    logger.info(f"Correct: {results['correct']}/{results['total']}")
    logger.info(f"Invalid responses: {results['invalid_responses']}")
    logger.info(f"Valid accuracy: {results['valid_accuracy']:.3f}")


if __name__ == "__main__":
    main()
