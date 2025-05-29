rollout_prompt = r"""I will provide a math problem along with an image of the problem. 

[Math Problem]

<math_problem> Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nQuestion: When the ant <image1> walks from home <image2> along the arrows $\rightarrow 3, \uparrow 3, \rightarrow 3, \uparrow 1$, he gets to the ladybird <image3>.\nWhich animal does the ant <image1> get to when he walks from home <image2> along the following arrows: $\rightarrow 2, \downarrow 2, \rightarrow 3, \uparrow 3, \rightarrow 2, \uparrow 2$?\n<image6>\n<image7\nChoices:\n(A) A\n(B) B\n(C) C\n(D) D\n(E) E. </math_problem>

Your task is to review the math problem and accompanying image of the problem in sequence, describing step-by-step what you see in the image in <perception> tags, reasoning step-by-step in <reasoning> tags, and finally providing the correct answer in <correct_answer> tags. You need to provide the analyzes and the correct answer in the following format:

[Perception]
<step_1>
...(step 1 of step-by-step perception)...
</step_1>
...
<step_n>
...(step n of step-by-step perception)...
</step_n>

[Reasoning]
<step_1>
...(step 1 of step-by-step reasoning)...
</step_1>
...
<step_n>
...(step n of step-by-step reasoning)...
</step_n>

<correct_answer>
...(correct answer deduced from step-by-step perception and reasoning)...
</correct_answer>

Now proceed with the task.
"""

rollout_system_prompt = r"""You are a math expert. I will provide a math problem along with an image of it. Your task is to review the math problem and the accompanying image in sequence, describing step-by-step what you see in the image in <perception> tags, reasoning step-by-step in <reasoning> tags, and finally providing the correct answer in <correct_answer> tags.

You need to provide the analysis and the correct answer in the following format:

[Perception]
<step_1>
...(Step 1 of step-by-step perception)...
</step_1>
...
<step_n>
...(Step n of step-by-step perception)...
</step_n>

[Reasoning]
<step_1>
...(Step 1 of step-by-step reasoning)...
</step_1>
...
<step_m>
...(Step m of step-by-step reasoning)...
</step_m>

<correct_answer>
...(Correct answer deduced from step-by-step perception and reasoning)...
</correct_answer>"""

rollout_user_prompt = r"""[Math Problem]

<math_problem>{{MATH_PROBLEM}}</math_problem>

Now begin the task.
"""