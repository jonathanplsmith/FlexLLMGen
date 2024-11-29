import dataclasses
from flexllmgen.utils import run_cmd
import tqdm

@dataclasses.dataclass
class LatencyResults:
    prefill_latency: float
    decode_latency: float

class ExpConfig:
    def __init__(self, prompt_len, gen_len, gbs, bls, percent, model_name="facebook/opt-66b"):
        self.stats_name = get_statsname(prompt_len, gen_len, gbs, bls, percent, model_name)
        self.prompt_len = prompt_len
        self.gen_len = gen_len
        self.model_name = model_name
        self.gbs = gbs # GPU batch size
        self.bls = bls # Block size
        self.percent = percent
    
    def get_cmd(self):
        return f"--model {self.model_name} --path _DUMMY_ --offload-dir /capstor/scratch/cscs/jsmith/flexllmgen_offload_dir/ --prompt-len {self.prompt_len} --gen-len {self.gen_len} --percent {self.percent[0]} {self.percent[1]} {self.percent[2]} {self.percent[3]} {self.percent[4]} {self.percent[5]} --gpu-batch-size {self.gbs} --num-gpu-batches {self.bls // self.gbs}"

def get_statsname(prompt_len, gen_len, gbs, bls, percentage, model_name):
    model_size = model_name.split('-')[-1]
    percent = ""
    for i in range(len(percentage)):
        percent += str(percentage[i]) + "-"
    filename = f"fo-{model_size}-gbs{gbs}-" \
               f"ngbs{bls // gbs}-" \
               f"prompt{prompt_len}-" \
               f"gen{gen_len}-percent-{percent}"
    filename += "gpu-cache"
    return filename

GiB = 1<<30

# Make sure we don't OOM
def invariant(prompt_len, gen_len, wr, kvr, ar, gbs, ngb):
    batch_size = gbs*ngb
    ratio_cpu_w = wr[1] / 100
    ratio_gpu_w = wr[0] / 100
    ratio_cpu_kv = kvr[1] / 100
    ratio_gpu_kv = kvr[0] / 100
    seqlen = prompt_len + gen_len
    kv_cache_total = 2 * batch_size * seqlen * 48  * 7168 * 2
    model_total = 65 * GiB
    gpu_total = kv_cache_total * ratio_gpu_kv + model_total * ratio_gpu_w
    cpu_total = kv_cache_total * ratio_cpu_kv + model_total * ratio_cpu_w
    return (gpu_total <= (90*GiB)) and (cpu_total <= (100*GiB))

#cases = [ExpConfig(prompt_len=512, gen_len=512, gbs=4, bls=4, percent=(50, 50, 50, 50, 100, 0))]
prompt_lens = [32, 128, 256]
gen_lens = [32, 256, 512]
weight_ratios = [(0, 100), (66, 34), (50, 50), (100, 0)]
kv_ratios = [(0, 100), (50, 50), (100, 0)]
actv_ratios = [(100, 0)]
gpu_block_sizes = [32, 64, 128, 256]
num_gpu_blocks = [3, 8, 12, 24]
cases = [
    ExpConfig(prompt_len=prompt_len, gen_len=gen_len, gbs=gbs, bls=gbs*ngb, percent=(wr[0], wr[1], kvr[0], kvr[1], ar[0], ar[1]), model_name="facebook/opt-30b")
    for prompt_len in prompt_lens
    for gen_len in gen_lens
    for wr in weight_ratios
    for kvr in kv_ratios
    for ar in actv_ratios
    for gbs in gpu_block_sizes
    for ngb in num_gpu_blocks
    if invariant(prompt_len, gen_len, wr, kvr, ar, gbs, ngb)
]

def get_filename(experiment):
    return experiment.stats_name

if __name__ == "__main__":
    for case in tqdm.tqdm(cases):
        cmd = f"python -m flexllmgen.flex_opt {case.get_cmd()} --debug fewer_batch"
        try:
            run_cmd(cmd)
        except Exception as e:
           print(e)
