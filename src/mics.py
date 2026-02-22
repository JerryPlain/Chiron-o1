from openai import OpenAI
import json
from utils import (
    encode_image, locate_img, get_correctness,
    gpt_forward, replace_image_references, qwenplus_forward, ds_forward, extract_first_two_steps,
    select_best_mentor, process_case_info, select_next_step, remove_phrases, extract_first_step
)
from qwenvl_forward import qwenvl_forward
from internvl_forward import internvl_forward
from prompt import EVALUATE_PROMPT, JUDGE_PROMPT, REASONING_PROMPT
from step import Step
import time

class MentorInternSearch:
    def __init__(self, args):
        """
        Initialize the Mentor-Intern Search process.
        Args:
            args: Command line arguments containing model lists, paths, max_depth, etc.
        """
        # Root node of the Step tree. Real reasoning steps are appended during search(...).
        self.root = Step(prefix_steps="") # Root node starts with empty prefix; actual reasoning steps are added as children during the search process.
        
        # Store runtime configuration so helper methods can access it consistently.
        self.args = args
        self.mentor_models = args.mentor_models
        self.intern_models = args.intern_models
        self.evaluator_model = args.evaluator_model 
        self.temperature1 = args.temperature1
        self.temperature2 = args.temperature2
        
        # Four OpenAI-compatible clients for different providers/endpoints.
        self.client1 = OpenAI(base_url=args.openai_base_url, api_key=args.openai_api_key)
        self.client2 = OpenAI(base_url=args.qwen_base_url, api_key=args.qwen_api_key)
        self.client3 = OpenAI(base_url=args.ds_base_url, api_key=args.ds_api_key)
        self.client4 = OpenAI(base_url=args.gemini_base_url, api_key=args.gemini_api_key)
        
        # Injected at search(...) time; used by local-model routes (qwen/internvl).
        self.model_set = None 
        

    def _call_model_forward(self, model_name, prompt, temperature, base64_images=None, img_paths=None, extensions=None):
        """
        Unified model router by model_name pattern.
        - Remote API routes: gpt/gemini/qwen72
        - Local routes: qwen2-vl / internvl
        Returns text output, or None on failure.
        """
        try:
            if 'gpt' in model_name:
                print("Calling gpt model!")
                return gpt_forward(self.client1, prompt, base64_images, temperature=temperature, model_name=model_name, extensions=extensions)
            elif 'gemini' in model_name:
                print("Calling gemini model!")
                return gpt_forward(self.client4, prompt, base64_images, temperature=temperature, model_name=model_name, extensions=extensions)
            elif '72' in model_name:
                print("Calling qwen model!")
                return qwenplus_forward(self.client2, prompt, base64_images, temperature=temperature, model_name=model_name, extensions=extensions)
            elif 'qwen' in model_name: 
                 response = qwenvl_forward(
                     self.model_set[model_name]['model'], 
                     self.model_set[model_name]['processor'],
                     prompt, 
                     img_paths, 
                     temperature=temperature 
                 )
                 return response
            elif 'internvl' in model_name: 
                 response = internvl_forward(
                      self.model_set[model_name]['model'], 
                      self.model_set[model_name]['processor'],
                      prompt, 
                      img_paths, 
                      temperature=temperature 
                 )
                 return response
            else:
                print(f"Warning: Model type for '{model_name}' not recognized for forwarding. Returning None.")
                return None
        except Exception as e:
            print(f"Error calling model {model_name} (Temp: {temperature}): {e}")
            return None


    def _generate_next_step_with_mentor(self, mentor_model_name, step, question, base64_images, img_paths, gt_answer, case_info, extensions):
        """
        Generates the next single reasoning step using a mentor model.
        Returns the full reasoning chain, including prefix, current step, and future steps.
        """
        # Next-step index is mainly for diagnostics/log readability.
        next_step_number = step.depth + 1
        
        # Reasoning prefix selected so far on the current path.
        reasoning_prefix = step.get_full_reasoning()

        # For remote API mentors, replace <image> placeholders with numbered references.
        # Local models keep the original case_info text.
        
        # mentors prompt, including models from gpt, gemini, and qwen72
        if 'gpt' in mentor_model_name  or 'gemini' in mentor_model_name or '72' in mentor_model_name:
            case_info = replace_image_references(case_info) # Replace <image> with Image 1, Image 2, etc. for remote API mentors that expect explicit references in the prompt.
            prompt = REASONING_PROMPT.format(
                reasoning_prefix=reasoning_prefix,
                question=question,
                gt_answer=gt_answer,
                case_info=case_info
            )
        # internvl often benefits from explicit <image> tags in the prompt, so we keep the original case_info with <image> placeholders for internvl mentors.
        else:
            prompt = REASONING_PROMPT.format(
                reasoning_prefix=reasoning_prefix,
                question=question,
                gt_answer=gt_answer,
                case_info=case_info
            )

        print("Starting to call mentor model!")
        time1 = time.time()

        # Ask mentor to continue from the existing reasoning prefix.
        response = self._call_model_forward(
            model_name=mentor_model_name, # Mentor model name determines which route and client to use in _call_model_forward.
            prompt=prompt,
            temperature=self.temperature1, # Use the first temperature for mentor generation to encourage more diverse reasoning steps.
            base64_images=base64_images, # Base64-encoded images for remote API mentors; local models will use img_paths directly.
            img_paths=img_paths,
            extensions=extensions
        )
        print(f"Mentor model call completed, time taken: {time.time() - time1} seconds")

        if response:
            return reasoning_prefix, response # return both the existing reasoning prefix and the mentor's full response, which includes the new step and future steps. The caller can then extract the new step and cache the full chain if needed.
        else:
            print(f" Failed to get response from {mentor_model_name} at generating step {next_step_number}.")
            return None

    def _evaluate_step_with_interns(self, step_to_evaluate, question, gt_answer, img_paths, mentors_scores):
        """
        Score one candidate step with all intern models.

        Protocol:
        - Each intern runs twice (temperature1 and temperature2)
        - Evaluator model judges each intern final answer as Yes/No
        - Final score = correct_count / (num_interns * 2)
        """

        # ==========================================================
        # [A] Build evaluation context from the candidate step.
        # ==========================================================
        print(f"Evaluating step (Depth {step_to_evaluate.depth}, Mentor: {step_to_evaluate.generated_by}) with interns...")
        reasoning_prefix = step_to_evaluate.get_full_reasoning()
        correct_count = 0
        num_intern_models = len(self.intern_models)
        total_evaluations = num_intern_models * 2

        if num_intern_models == 0:
            print("Warning: No intern models specified for evaluation. Returning score 0.")
            return 0.0

        # ==========================================================
        # [B] Run each intern with two temperatures and collect outputs.
        # ==========================================================
        for intern_model_name in self.intern_models:
            # InternVL prompt usually needs explicit image placeholders.
            if 'internvl' in intern_model_name:
                image_prefix = ""
                for i, img_path in enumerate(img_paths, 1):
                    image_prefix += f"Image {i}: <image>\n"

                question = image_prefix + question
                intern_prompt = EVALUATE_PROMPT.format(
                    question=question,
                    reasoning_prefix=reasoning_prefix
                )
            else:
                intern_prompt = EVALUATE_PROMPT.format(
                    question=question,
                    reasoning_prefix=reasoning_prefix
                )

            # Two passes per intern to reduce sampling variance.
            for temp in [self.temperature1, self.temperature2]:
                print(f"  Running intern: {intern_model_name} (Temp: {temp})")
                time1 = time.time()

                intern_response = self._call_model_forward(
                    model_name=intern_model_name,
                    prompt=intern_prompt,
                    temperature=temp,
                    img_paths=img_paths
                )
                print(f"Intern model call completed, time taken: {time.time() - time1} seconds")

                if not intern_response:
                    print(f"Failed to get response from intern {intern_model_name} (Temp: {temp})")
                    continue

                # Require the expected marker to parse the final answer safely.
                if '### The final answer is:' not in intern_response:
                    print(f"Warning: Intern {intern_model_name} (Temp: {temp}) response did not contain '### The final answer is:'.")
                    continue

                model_answer = intern_response.split('### The final answer is:')[-1].strip()
                if not model_answer:
                    print(f"Warning: Intern {intern_model_name} (Temp: {temp}) provided empty final answer.")
                    continue

                # ==========================================================
                # [C] Ask evaluator model to judge correctness for this run.
                # ==========================================================
                judge_prompt = JUDGE_PROMPT.format(
                    question=question,
                    model_answer=model_answer,
                    gt_answer=gt_answer
                )

                time2 = time.time()
                judge_output = ds_forward(self.client3, judge_prompt, temperature=0.9, model_name=self.evaluator_model)
                print(f"Evaluator model call completed, time taken: {time.time() - time2} seconds")
                if judge_output:
                    is_correct = get_correctness(judge_output)
                    if is_correct == 1:
                        correct_count += 1
                else:
                    print(f" Failed to get judgment from evaluator {self.evaluator_model}")

        # ==========================================================
        # [D] Aggregate all judgments into one candidate-step score.
        # ==========================================================
        score = correct_count / total_evaluations if total_evaluations > 0 else 0.0
        print(f"  Step Score (Correct/Total Evaluations): {correct_count}/{total_evaluations} = {score:.2f}")

        return score
    
    # Main search entry for one sample: mentor generation -> intern evaluation -> path selection.
    def search(self, data, model_set, search_file, failed_search_file):
        """
        Execute mentor-intern search for one sample and write result logs immediately.
        """
        self.model_set = model_set  # Local model handles used by _call_model_forward.

        # ==========================================================
        # [A] Parse sample inputs and prepare model-ready fields.
        # - Extract question and gold answer
        # - Build structured case_info text
        # - Resolve image paths and encode images
        # Any failure here is logged to failed_search_file and exits early.
        # ==========================================================
        question = data["messages"][0]['content']
        question = question.replace('**Question:**', ' ').strip()
        
        case_info = process_case_info(data) # Format case information into a structured string for prompting. e.g. "Chief complaint: ... Age: ... Gender: ... Captions: ..."

        gt_answer = data["messages"][1]['content']
        gt_answer = gt_answer.replace('**Correct Answer:**', ' ').strip()

        # Resolve image paths first, then base64-encode each image once.
        try:
            img_paths, extensions = locate_img(data, self.args) 
            base64_images = [encode_image(img_path) for img_path in img_paths]
        except ValueError as e:
            print(f"Error encoding image for data item: {e}. Skipping item.")
            failed_data = data.copy()
            failed_data['error'] = str(e)
            failed_search_file.write(json.dumps(failed_data) + "\n")
            failed_search_file.flush()
            return
        except FileNotFoundError as e:
             print(f"Error finding image for data item: {e}. Skipping item.")
             failed_data = data.copy()
             failed_data['error'] = str(e)
             failed_search_file.write(json.dumps(failed_data) + "\n")
             failed_search_file.flush()
             return

        print(f"\n--- Starting Search for Question ID: {data.get('rid', 'N/A')} ---")

        # ==========================================================
        # [B] Initialize search state for this sample.
        # current_step is the active node on the selected path.
        # ==========================================================
        initial_text = "Let's think about how to solve this problem clearly and reasonably step by step."
        self.root.text = initial_text
        
        # current_step starts at root; as we generate and select child steps, we will move current_step down the tree to build the reasoning path.
        current_step = self.root

        # mentors_scores: mentor -> [score_at_depth_0, score_at_depth_1, ...]
        mentors_scores = {mentor: [] for mentor in self.mentor_models}
        
        # reasoning_chains: mentor -> latest full reasoning text generated by that mentor
        reasoning_chains = {} 
        
        # previous_mentors: mentors selected on the final chosen path
        previous_mentors = [] 

        # ==========================================================
        # [C] Depth-wise iterative search.
        # For each depth:
        # 1) each mentor proposes one candidate step
        # 2) interns evaluate that candidate and return a score
        # 3) pick one child to continue to the next depth
        # ==========================================================
        for depth in range(self.args.max_depth):
            print(f"\n-- Depth {depth} (Generating Step {depth + 1}) --")

            # Candidate child nodes generated at this depth.
            generated_children_for_step = []
            
            # Mentors that achieved perfect score (1.0) at this depth.
            full_score_mentors = []
            
            # Early-fail flag: remains True only if every mentor scores 0 at this depth.
            all_zero_score = True
            
            for mentor_model in self.mentor_models:
                print(f"Generating step with mentor: {mentor_model}")
                
                # Root-layer strategy: extract first two steps to bootstrap faster.
                # _generate_next_step_with_mentor will return 
                # - prefix_reasoning: the existing reasoning chain up to current_step, which can be reused for the same mentor if it continues in the next iteration (common at root layer). it is from step.get_full_reasoning()
                # - response: the mentor's full response including the new step and future steps. We will extract the new step for evaluation, and cache the full response for potential reuse if the same mentor
                if current_step.parent is None:
                    prefix_reasoning, suffix_reasoning = self._generate_next_step_with_mentor(
                    mentor_model, current_step, question, base64_images, img_paths, gt_answer, case_info, extensions
                )
                    # Cache the full reasoning chain from this mentor to reuse if the same mentor continues in the next iteration, which is common at the root layer.
                    complete_reasoning = prefix_reasoning + "\n" + suffix_reasoning
                    reasoning_chains[mentor_model] = complete_reasoning  
                    
                    new_step = extract_first_two_steps(suffix_reasoning)    
                    new_step = "### " + new_step
                else: 
                    # If same mentor continues, reuse cached chain to avoid extra calls.
                    if current_step.generated_by == mentor_model:
                        try:
                            new_step = reasoning_chains[mentor_model].split("###")[depth + 2].strip()
                            new_step = "### " + new_step
                        except IndexError:
                            new_step = ""

                    else:
                        # For a different mentor, regenerate continuation from current prefix.
                        prefix_reasoning, suffix_reasoning = self._generate_next_step_with_mentor(
                        mentor_model, current_step, question, base64_images, img_paths, gt_answer, case_info, extensions
                    )

                        complete_reasoning = prefix_reasoning + "\n" + suffix_reasoning
                        reasoning_chains[mentor_model] = complete_reasoning  

                        # extract first new step for evaluation; the full chain is cached for potential reuse if this mentor continues in the next iteration.
                        new_step = extract_first_step(suffix_reasoning) 
                        new_step = "### " + new_step
                
                # intern scoring needs the full chain including the new step and future steps, so we construct a temporary step with the full reasoning for evaluation. 
                # The actual child added to the tree will only have the new step text, and its prefix_steps will be the current_step's full reasoning.
                # current_step.text is the full reasoning up to current_step, which serves as the prefix for the new step. The new step text is just the newly generated step from the mentor.
                temp_step = Step(step_text=new_step, prefix_steps=current_step.text, parent=current_step, generated_by=mentor_model)
                
                # Evaluate this candidate step with all interns and get a score.
                score = self._evaluate_step_with_interns(
                    temp_step, question, gt_answer, img_paths, mentors_scores
                )
                # Record this mentor's score at the current depth.
                mentors_scores[mentor_model].append(score)

                # Update the all_zero_score flag for early termination check. If any mentor scores above 0, we can continue; if all remain 0 after this loop, we will stop early.
                if score > 0:
                    all_zero_score = False
                actual_child = current_step.add_child_step(step_text=new_step, score=score, generated_by=mentor_model)
                
                # if actual_child is None, it means the new_step was None and we skipped adding this child. 
                # In that case, we should not consider this mentor's output for path selection or early termination, since it is effectively a failure to generate a valid step.
                if actual_child: 
                    generated_children_for_step.append(actual_child)
                    if score == 1.0:
                        full_score_mentors.append(mentor_model)  

            # ==========================================================
            # [D] Early termination checks for current depth.
            # - Success: any mentor reaches score 1.0
            # - Failure: all mentors score 0
            # ==========================================================
            # If any mentor reaches full score, stop early and write success result.
            if full_score_mentors:
                if len(full_score_mentors) == 1:
                    best_mentor = full_score_mentors[0] # If only one mentor achieved full score, we select that mentor's chain as the final reasoning.
                else:
                    best_mentor = select_best_mentor(mentors_scores, depth) # If multiple mentors achieved full score, we can use a tie-breaking strategy to select the best mentor among them. Here we use select_best_mentor to pick the mentor with the highest score at this depth, which indicates stronger consensus from interns.
                print(f"Full score achieved by {len(full_score_mentors)} mentors. Selected mentor: {best_mentor}")
                full_reasoning = reasoning_chains[best_mentor]
                
                result_data = {
                    'rid': data.get('rid', 'N/A'),
                    'images': data['images'],
                    'question': question,
                    'gt_answer': gt_answer,
                    'reasoning': full_reasoning,
                    'scores': mentors_scores,  
                    'final_depth': str(depth + 1),  
                    'generated_by': previous_mentors + [best_mentor],
                    'search_id': '1'
                }
                
                search_file.write(json.dumps(result_data) + "\n")
                search_file.flush()
                print(f"Saved successful result with full score for {data.get('rid', 'N/A')}")
                return  

            # If all scores are zero at this depth, write failure and exit early.
            if all_zero_score:
                print("All mentor scores are zero. Stopping search and recording as failure.")
                failed_data = data.copy()
                failed_data['error'] = "All mentor scores are zero."
                failed_search_file.write(json.dumps(failed_data) + "\n")
                failed_search_file.flush()
                return

            # Normal flow: pick the next active node and continue search.
            if generated_children_for_step:
                current_step = select_next_step(generated_children_for_step, previous_mentors)
                previous_mentors.append(current_step.generated_by)
                print(f" Step (Depth {depth}) best child score: {current_step.score:.2f} (Mentor: {current_step.generated_by})")
            else:
                print("No valid steps generated. Stopping search.")
                break

            print("########################################################")

        # ==========================================================
        # [E] Reached max_depth without early stop.
        # ==========================================================
        print(f"\n--- Search Completed (Max Depth Reached) ---")
        
        # ==========================================================
        # [F] Persist final output.
        # Write success if a valid mentor chain exists; otherwise write failure.
        # ==========================================================
        # By default, use the final selected mentor's chain as output reasoning.
        if current_step.score >= 0:  
            final_mentor = current_step.generated_by
        else :
            final_mentor = "No valid mentor!"

        if final_mentor in reasoning_chains:
            final_reasoning = reasoning_chains[final_mentor]
            result_data = {
                'rid': data.get('rid', 'N/A'),
                'images': data['images'],
                'question': question,
                'gt_answer': gt_answer,
                'reasoning': final_reasoning,
                'scores': mentors_scores,  
                'final_depth': str(depth + 1),  
                'generated_by': previous_mentors,  # {current_step.get_step_path()[i].generated_by for i in range(len(current_step.get_step_path()))}
                'search_id': '0'
            }
                    
            search_file.write(json.dumps(result_data) + "\n")
            search_file.flush()
            print(f"Saved result for {data.get('rid', 'N/A')}")
        else:
            failed_data = data.copy()
            failed_data['error'] = "Search failed to find a valid reasoning path."
            failed_search_file.write(json.dumps(failed_data) + "\n")
            failed_search_file.flush()
            print(f"Saved failed search log for {data.get('rid', 'N/A')}")

        print(f"--- Search Finished for Question ID: {data.get('rid', 'N/A')} ---")
