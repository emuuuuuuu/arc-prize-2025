import uuid, json
from tqdm.auto import tqdm
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

MODEL_PATH   = "/kaggle/input/unslothqwen3-8b-unsloth-bnb-4bit"
ADAPTER_PATH = "/kaggle/input/ds_qwen_8b_adapter2/pytorch/default/1"
TEST_JSON    = "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json"
SUBMIT_FILE  = "submission.json"

MAX_LEN = 4096
BATCH_SIZE = 240

lora_req = LoRARequest("arc_adapter", 1, ADAPTER_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
    model=MODEL_PATH,
    quantization="bitsandbytes",
    pipeline_parallel_size=4,
    tensor_parallel_size=1,
    enable_lora=True, fully_sharded_loras=True,
    gpu_memory_utilization=0.92,
    max_model_len=MAX_LEN,
    max_num_batched_tokens=16384,
    max_num_seqs=BATCH_SIZE,
    enforce_eager=False, 
    max_seq_len_to_capture=4096,
    tokenizer_pool_size=4,
    tokenizer_mode="auto",
    kv_cache_dtype="fp8_e5m2",
    trust_remote_code=True,
))

params = SamplingParams(temperature=0, max_tokens=2048)

SYS = {
    "role": "system",
    "content": "You are a world-class puzzle solver with exceptional pattern recognition skills. You will be given several examples of input-output pairs of matrices which relate via some pattern, like in a IQ matrix puzzle. You should be able to find a rule/transformation that lets you find the output matrix given the input matrix of all the examples in this task. Then based on the pattern you find you will apply that pattern transformation on the test input matrix, go through the problem step by step and finally give your solution output matrix enclosed in the special token ```."
}
END = (
    "Now try to see the pattern, and apply the transformation step by step on this input matrix and eventually give your answer output matrix. Do NOT code your solution, "
    "write it in english and show your work with intermediate matrices."
)
def g2s(g): return "\n".join("".join(map(str,r)) for r in g)
def build_prompt(chal):
    msgs=[SYS]
    for i,tr in enumerate(chal["train"]):
        msgs.append({"role":"user","content":
            f"Example {i+1}:\nInput:\n{g2s(tr['input'])}\n"
            f"Output:\n{g2s(tr['output'])}\n"})
    msgs += [
        {"role":"user","content":
         f"Here is the test input:\nInput:\n{g2s(chal['test'][0]['input'])}"},
        END]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def parse(txt):
    try:
        inner=txt.split("```")[1].strip()
        return [[int(c) for c in row] for row in inner.split("\n") if row]
    except Exception:
        return [[0]]

async def gen_one(prompt):
    rid = uuid.uuid4().hex
    final = ""
    async for out in engine.generate(
        prompt,
        params,
        request_id=rid,
        lora_request=lora_req,
    ):
        if out.outputs:
            final = out.outputs[0].text
        if out.finished:
            break
    return final

async def run_all():
    with open(TEST_JSON) as f:
        tests = json.load(f)

    # filter by tokenized length
    max_len = MAX_LEN  # must match engine max_model_len
    filtered = []
    for tid, chal in tests.items():
        prompt = build_prompt(chal)
        if len(tokenizer(prompt, add_special_tokens=False).input_ids) <= max_len:
            filtered.append((tid, prompt))
        else:
            print(f"Skipping {tid} (len {len(tokenizer(prompt, add_special_tokens=False).input_ids)})")
    items = filtered

    submission = {}
    for i in range(0, len(items), BATCH_SIZE):
        chunk = items[i:i+BATCH_SIZE]

        # 1) launch requests and keep their streams
        launched = []
        for tid, prompt in chunk:
            rid = uuid.uuid4().hex
            stream = await engine.add_request(
                rid, prompt, params, lora_request=lora_req
            )  # async generator/stream
            launched.append((tid, stream))

        # 2) drain each stream to completion
        for tid, stream in launched:
            final = ""
            async for out in stream:
                if out.outputs:
                    final = out.outputs[0].text
                if out.finished:
                    break
            print(f"ARC-ID: {tid}\n{final}\n{'='*60}")
            submission[f"{tid}_0"] = parse(final)

    return submission

submission = await run_all()

with open(SUBMIT_FILE,"w") as f:
    json.dump(submission,f,indent=4)
print("Saved →", SUBMIT_FILE)