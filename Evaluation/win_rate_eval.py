import os
import json
import asyncio
import argparse
import random
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

EVAL_PROMPT = """You are an impartial judge evaluating two different AI answers based on the ground truth.

Question: {question}
Ground Truth: {ground_truth}

Answer A:
{ans_A}

Answer B:
{ans_B}

Please compare both answers against the Ground Truth. Determine which answer is more accurate, comprehensive, and faithful to the ground truth.
If one is clearly better, choose that model. If they are equally good or equally bad, choose TIE.

Provide a brief reasoning, and YOU MUST END YOUR RESPONSE WITH EXACTLY ONE OF THESE TAGS:
[MODEL_A_WINS], [MODEL_B_WINS], or [TIE].
"""

async def evaluate_pair(llm, item_A, item_B):
    question = item_A["question"]
    ground_truth = item_A["ground_truth"]
    
    ans_1 = item_A["generated_answer"]
    ans_2 = item_B["generated_answer"]
    
    # Randomize position to avoid bias
    swapped = random.choice([True, False])
    if swapped:
        ans_A, ans_B = ans_2, ans_1
    else:
        ans_A, ans_B = ans_1, ans_2
        
    prompt = EVAL_PROMPT.format(question=question, ground_truth=ground_truth, ans_A=ans_A, ans_B=ans_B)
    
    try:
        response = await llm.ainvoke([SystemMessage(content="You are a strict, objective AI judge."), HumanMessage(content=prompt)])
        content = response.content
        
        # Parse result
        if "[MODEL_A_WINS]" in content:
            winner = "B" if swapped else "A"
        elif "[MODEL_B_WINS]" in content:
            winner = "A" if swapped else "B"
        else:
            winner = "TIE"
            
        return winner, content
    except Exception as e:
        print(f"Error evaluating {item_A['id']}: {e}")
        return "ERROR", str(e)

async def main(args):
    with open(args.file_a, "r") as f:
        data_a = json.load(f)
    with open(args.file_b, "r") as f:
        data_b = json.load(f)
        
    dict_a = {item["id"]: item for item in data_a}
    dict_b = {item["id"]: item for item in data_b}
    
    common_ids = set(dict_a.keys()) & set(dict_b.keys())
    
    if args.question_type:
        common_ids = {qid for qid in common_ids if dict_a[qid].get("question_type") == args.question_type}
        print(f"Filtered for question type '{args.question_type}'. Remaining: {len(common_ids)}")
    else:
        print(f"Found {len(common_ids)} common questions.")

    if not common_ids:
        print("No questions to evaluate.")
        return
    
    llm = ChatOpenAI(model=args.model, temperature=0.1, api_key=os.environ.get("OPENAI_API_KEY"))
    
    sem = asyncio.Semaphore(10)
    
    wins_a = 0
    wins_b = 0
    ties = 0
    
    random.seed()
    common_ids = list(common_ids)
    random.shuffle(common_ids)
    
    async def task(qid):
        async with sem:
            winner, reason = await evaluate_pair(llm, dict_a[qid], dict_b[qid])
            return qid, winner, reason
            
    tasks = [task(qid) for qid in common_ids]
    
    results = []
    completed = 0
    total = len(common_ids)
    for future in asyncio.as_completed(tasks):
        qid, winner, reason = await future
        results.append({"id": qid, "winner": winner, "reason": reason})
        
        if winner == "A": wins_a += 1
        elif winner == "B": wins_b += 1
        elif winner == "TIE": ties += 1
        
        completed += 1
        print(f"[{completed}/{total}] Winner: {winner} | Score -> A: {wins_a}, B: {wins_b}, TIE: {ties}")
        
    print("\n" + "="*40)
    print(f"FINAL WIN RATE RESULTS (Type: {args.question_type if args.question_type else 'All'})")
    print(f"Model A ( {os.path.basename(args.file_a)} ) Wins : {wins_a} ({(wins_a/total)*100:.1f}%)")
    print(f"Model B ( {os.path.basename(args.file_b)} ) Wins : {wins_b} ({(wins_b/total)*100:.1f}%)")
    print(f"Ties : {ties} ({(ties/total)*100:.1f}%)")
    print("="*40)
    
    with open(args.output, "w") as f:
        json.dump({
            "model_A_file": args.file_a,
            "model_B_file": args.file_b,
            "question_type": args.question_type,
            "wins_A": wins_a,
            "wins_B": wins_b,
            "ties": ties,
            "details": results
        }, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_a", type=str, required=True)
    parser.add_argument("--file_b", type=str, required=True)
    parser.add_argument("--output", type=str, default="win_rate_results.json")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--question_type", type=str, default=None)
    args = parser.parse_args()
    asyncio.run(main(args))
