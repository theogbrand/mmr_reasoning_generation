def parse_perception_reasoning_output(text):
    """
    Parse text that contains perception steps, reasoning steps, and a correct answer.
    
    Args:
        text (str): The text to parse
        
    Returns:
        dict: Dictionary with 'perception_steps', 'reasoning_steps', and 'llm_answer'
        
    Raises:
        ValueError: If the text doesn't contain all required sections
    """
    import re
    
    # Initialize the result dictionary
    result = {
        'perception_steps': [],
        'reasoning_steps': [],
        'llm_answer': None
    }
    
    # Extract perception steps
    perception_pattern = r'\[Perception\](.*?)(?=\[Reasoning\]|\Z)'
    perception_match = re.search(perception_pattern, text, re.DOTALL)
    
    if not perception_match:
        raise ValueError("Could not find Perception section")
    
    perception_text = perception_match.group(1).strip()
    step_pattern = r'<step_(\d+)>(.*?)</step_\1>'
    perception_steps = re.findall(step_pattern, perception_text, re.DOTALL)
    
    if not perception_steps:
        raise ValueError("Could not find any perception steps")
    
    # Sort by step number and extract content
    perception_steps.sort(key=lambda x: int(x[0]))
    result['perception_steps'] = [step[1].strip() for step in perception_steps]
    
    # Extract reasoning steps
    reasoning_pattern = r'\[Reasoning\](.*?)(?=<correct_answer>|\Z)'
    reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
    
    if not reasoning_match:
        raise ValueError("Could not find Reasoning section")
    
    reasoning_text = reasoning_match.group(1).strip()
    reasoning_steps = re.findall(step_pattern, reasoning_text, re.DOTALL)
    
    if not reasoning_steps:
        raise ValueError("Could not find any reasoning steps")
    
    # Sort by step number and extract content
    reasoning_steps.sort(key=lambda x: int(x[0]))
    result['reasoning_steps'] = [step[1].strip() for step in reasoning_steps]
    
    # Extract correct answer
    answer_pattern = r'<correct_answer>(.*?)</correct_answer>'
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    
    if not answer_match:
        raise ValueError("Could not find correct answer")
    
    result['llm_answer'] = answer_match.group(1).strip()
    
    # Final validation to ensure we have all components
    if not result['perception_steps'] or not result['reasoning_steps'] or not result['llm_answer']:
        raise ValueError("Missing one or more required components")
    
    return result

if __name__ == "__main__":
    import json
    import os
    print("Parsing perception reasoning output")
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct absolute paths for input and output files
    input_file = os.path.join(script_dir, "test_raw_reasoning_traces.json")
    output_file = os.path.join(script_dir, "test_output_parsed.jsonl")
    
    with open(input_file, "r") as f:
        traces = json.load(f)
    for i, trace in enumerate(traces["traces"]):
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                result = parse_perception_reasoning_output(trace)
                with open(output_file, "a") as f:
                    f.write(json.dumps(result) + "\n")
                break
            except Exception as e:
                print(f"Error parsing trace {i}, {trace[:100]}...: {str(e)}")
                retry_count += 1
                if retry_count == max_retries:
                    print(f"Failed to parse trace {i} after {max_retries} attempts: {str(e)}")
                    break
                print(f"Retry {retry_count} for trace {i}")
        if i > 10:
            break
    print(f"Done parsing {i+1} traces of perception reasoning output")