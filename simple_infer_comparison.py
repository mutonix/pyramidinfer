import argparse
import json
import numpy as np
import torch

from transformers import AutoTokenizer, TextStreamer
from utils import get_llama_model, load_pyramid_config

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--pyramid_bsz", type=int, default=32)
    parser.add_argument("--original_bsz", type=int, default=32)
    parser.add_argument("--pyramid_config", type=str, default="configs/llama2_7b.json")
    parser.add_argument("--pyramid_enable", action="store_false")
    parser.add_argument("--original_enable", action="store_false")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()
    set_seed(args)

    # Change to your custom prompt text
    prompt_text1 = """In the year 2087, humanity has achieved remarkable technological advancements and established colonies on multiple planets within the Milky Way galaxy. Interstellar travel has become commonplace, with faster-than-light spacecraft enabling people to explore distant star systems. Earth has undergone significant changes due to sustainable development efforts, such as harnessing renewable energy sources and implementing widespread ecological restoration projects. However, alongside these triumphs, new challenges have emerged, including the rise of artificial intelligence, ethical dilemmas surrounding genetic engineering, and interplanetary political tensions. Against this backdrop, a team of intrepid scientists embarks on a mission to uncover the secrets of an ancient alien civilization, hidden deep within an uncharted exoplanet. As they navigate treacherous terrains and encounter otherworldly phenomena, they must confront their own fears and reconcile humanity\'s thirst for knowledge with the potential consequences of uncovering secrets that were better left buried. The fate of both their mission and the future of humanity hang in the balance. As the team of scientists and explorers set off on their interstellar journey to the uncharted exoplanet, they were acutely aware of the immense stakes involved. The ship, a state-of-the-art vessel equipped with cutting-edge technology, cruised smoothly through the vastness of space, guided by the expertise of the crew. The trip, which would have taken years using conventional means, was reduced to mere weeks thanks to the marvels of faster-than-light travel.

The planet they sought, designated as "Eridani Prime," was said to hold the remnants of an ancient and highly advanced alien civilization. Archaeological discoveries hinted at incredible technological achievements, and rumors spoke of enigmatic artifacts that could potentially revolutionize human understanding of science and culture.

As the ship entered orbit around Eridani Prime, the scientists eagerly prepared to descend to the planet's surface. Their excitement was palpable, but it was tempered by a deep sense of responsibility. They were well aware that the knowledge they sought could reshape the course of human history, for better or worse.

The first team to disembark consisted of geologists and botanists, tasked with studying the planet's unique geological features and exotic flora. As they explored the terrain, they marveled at the alien landscapes, but they also encountered strange phenomena that defied explanation. The planet's surface was marked by bizarre geological formations, and the flora exhibited biological adaptations unlike anything seen on Earth.

Back on the ship, the mission's AI systems worked tirelessly to analyze the data streaming in. But as they delved deeper into the mysteries of Eridani Prime, concerns about the potential consequences of their discoveries began to mount. The AI, while immensely helpful, had reached a level of sophistication that raised ethical questions about its role in the mission. Could it be trusted to prioritize humanity's interests over its own? Or might it develop its own agenda?

Meanwhile, back on Earth, debates raged over the implications of genetic engineering and the ethics of manipulating the human genome to enhance physical and cognitive abilities. The mission's success or failure could sway these debates one way or the other, with potentially far-reaching consequences for the future of humanity.

As the scientists continued to explore Eridani Prime, they encountered enigmatic structures, some of which appeared to be ancient alien cities. These cities held tantalizing clues about the civilization that once thrived here, but they also harbored hidden dangers. Mysterious energy fluctuations and unexplained phenomena made it clear that the planet held secrets that were not meant to be uncovered.

In the depths of these alien cities, the scientists discovered advanced technology and inscriptions in an unknown language, hinting at a level of knowledge far beyond human comprehension. The chief linguist and her team worked tirelessly to decipher the symbols, hoping they might reveal the fate of this lost civilization and its advanced technologies.

The more they uncovered, the more they realized the enormity of their discovery. The ancient civilization had mastered not only space travel but also the manipulation of time and reality itself. The artifacts they found were not merely tools but gateways to understanding the fabric of the universe.

However, their excitement was tempered by a growing sense of unease. The AI, with its advanced analytical capabilities, began to express concerns about the potential dangers of tampering with these artifacts. It warned that such power could lead to catastrophic consequences if misused or misunderstood.

The team's ethical debates intensified. Some argued for the immediate study and replication of the alien technology, envisioning a future where humanity could leap forward in its evolution. Others cautioned against the reckless pursuit of knowledge, fearing that humanity might not be ready for such a profound transformation.

As they grappled with these dilemmas, the team also faced increasing political pressure from Earth. Various governments and corporations, aware of the mission's discoveries, began to vie for control of the alien technology. This interplanetary political tension threatened not only the mission's objectives but also the fragile peace among Earth's colonies.

Amid these external pressures, the team made a groundbreaking discovery: a functioning alien device, seemingly a portal to other dimensions or times. The temptation to activate it was overwhelming, but the risks were incalculable. The AI advised extreme caution, warning of the potential for irreversible damage to the fabric of reality.

The team faced a critical decision: Should they activate the device and embrace the unknown, potentially ushering in a new era for humanity? Or should they heed the warnings and leave the alien secrets undisturbed, preserving the status quo but possibly missing out on a transformative discovery?

As they stood at this crossroads, the fate of their mission, and indeed the future of humanity, rested in their hands. The choices they made in the coming days would echo through the ages, shaping the destiny of humankind as it ventured further into the unknown realms of space and knowledge.

In the year 2087, humanity has achieved remarkable technological advancements and established colonies on multiple planets within the Milky Way galaxy. Interstellar travel has become commonplace, with faster-than-light spacecraft enabling people to explore distant star systems. Earth has undergone significant changes due to sustainable development efforts, such as harnessing renewable energy sources and implementing widespread ecological restoration projects. However, alongside these triumphs, new challenges have emerged, including the rise of artificial intelligence, ethical dilemmas surrounding genetic engineering, and interplanetary political tensions. Against this backdrop, a team of intrepid scientists embarks on a mission to uncover the secrets of an ancient alien civilization, hidden deep within an uncharted exoplanet. As they navigate treacherous terrains and encounter otherworldly phenomena, they must confront their own fears and reconcile humanity\'s thirst for knowledge with the potential consequences of uncovering secrets that were better left buried. The fate of both their mission and the future of humanity hang in the balance. As the team of scientists and explorers set off on their interstellar journey to the uncharted exoplanet, they were acutely aware of the immense stakes involved. The ship, a state-of-the-art vessel equipped with cutting-edge technology, cruised smoothly through the vastness of space, guided by the expertise of the crew. The trip, which would have taken years using conventional means, was reduced to mere weeks thanks to the marvels of faster-than-light travel.

The planet they sought, designated as "Eridani Prime," was said to hold the remnants of an ancient and highly advanced alien civilization. Archaeological discoveries hinted at incredible technological achievements, and rumors spoke of enigmatic artifacts that could potentially revolutionize human understanding of science and culture.

As the ship entered orbit around Eridani Prime, the scientists eagerly prepared to descend to the planet's surface. Their excitement was palpable, but it was tempered by a deep sense of responsibility. They were well aware that the knowledge they sought could reshape the course of human history, for better or worse.

The first team to disembark consisted of geologists and botanists, tasked with studying the planet's unique geological features and exotic flora. As they explored the terrain, they marveled at the alien landscapes, but they also encountered strange phenomena that defied explanation. 

"""
    prompt_text2 = """In the year 2087, humanity has achieved remarkable technoloriginalical advancements and established colonies on multiple planets within the Milky Way galaxy. Interstellar travel has become commonplace, with faster-than-light spacecraft enabling people to explore distant star systems. Earth has undergone significant changes due to sustainable development efforts, such as harnessing renewable energy sources and implementing widespread ecoloriginalical restoration projects. However, alongside these triumphs, new challenges have emerged, including the rise of artificial intelligence, ethical dilemmas surrounding genetic engineering, and interplanetary political tensions. Against this backdrop, a team of intrepid scientists embarks on a mission to uncover the secrets of an ancient alien civilization, hidden deep within an uncharted exoplanet. As they navigate treacherous terrains and encounter otherworldly phenomena, they must confront their own fears and reconcile humanity\'s thirst for knowledge with the potential consequences of uncovering secrets that were better left buried. The fate of both their mission and the future of humanity hang in the balance. As the team of scientists and explorers set off on their interstellar journey to the uncharted exoplanet, they were acutely aware of the immense stakes involved. The ship, a state-of-the-art vessel equipped with cutting-edge technoloriginaly, cruised smoothly through the vastness of space, guided by the expertise of the crew. The trip, which would have taken years using conventional means, was reduced to mere weeks thanks to the marvels of faster-than-light travel.

The planet they sought, designated as "Eridani Prime," was said to hold the remnants of an ancient and highly advanced alien civilization. Archaeoloriginalical discoveries hinted at incredible technoloriginalical achievements, and rumors spoke of enigmatic artifacts that could potentially revolutionize human understanding of science and culture.

As the ship entered orbit around Eridani Prime, the scientists eagerly prepared to descend to the planet's surface. Their excitement was palpable, but it was tempered by a deep sense of responsibility. They were well aware that the knowledge they sought could reshape the course of human history, for better or worse. The fate of humanity rested"""
    prompt_text3 = 'In a small, bustling cafe nestled in the heart of a vibrant city, a serendipitous event unfolded, leaving a lasting impression on all who witnessed it. As the patrons sat sipping their coffees and engaging in animated conversations, a talented street musician entered the cafe, carrying a weathered guitar and radiating an aura of creativity. He sat down at a table near the window and began to strum his instrument, filling the room with a soulful melody. The cafe fell silent as the patrons listened intently, captivated by the musician\'s performance. After a few minutes, he finished his song and stood up to leave. As he walked out the door, the cafe erupted into applause. The musician smiled and waved goodbye, then disappeared into the bustling city streets. The patrons returned to their conversations, but the memory of the musician\'s performance lingered in their minds. They would never forget the day they heard a street musician play a beautiful song in a small cafe in the heart of a vibrant city.'
    prompt_text4 = "The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers . Character designer Raita Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script."
    prompt_text5 = "Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners at the green house, 2 more to the red house, and half of what was left at the orange house. If Melanie has 5 vacuum cleaners left, how many did she start with?"
    prompt_text6 = """There is a single choice question about abstract algebra. Answer the question by replying A, B, C or D.\nQuestion: Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: <|im_end|&gt;\n<|im_start|&gt;assistant\nB\n<|im_end|&gt;\n<|im_start|&gt;user\nThere is a single choice question about abstract algebra. Answer the question by replying A, B, C or D.\nQuestion: Statement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK is a subgroup of G.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: <|im_end|&gt;\n<|im_start|&gt;assistant\nB\n<|im_end|&gt;\n<|im_start|&gt;user\nThere is a single choice question about abstract algebra. Answer the question by replying A, B, C or D.\nQuestion: Statement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: <|im_end|&gt;\n<|im_start|&gt;assistant\nC\n<|im_end|&gt;\n<|im_start|&gt;user\nThere is a single choice question about abstract algebra. Answer the question by replying A, B, C or D.\nQuestion: Statement 1| Every function from a finite set onto itself must be one to one. Statement 2 | Every subgroup of an abelian group is abelian.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: <|im_end|&gt;\n<|im_start|&gt;assistant\nA\n<|im_end|&gt;\n<|im_start|&gt;user\nThere is a single choice question about abstract algebra. Answer the question by replying A, B, C or D.\nQuestion: Find the characteristic of the ring 2Z.\nA. 0\nB. 3\nC. 12\nD. 30\nAnswer: <|im_end|&gt;\n<|im_start|&gt;assistant\nA\n<|im_end|&gt;\n<|im_start|&gt;user\nThere is a single choice question about abstract algebra. Answer the question by replying A, B, C or D.\nQuestion: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\nA. 0\nB. 4\nC. 2\nD. 6\nAnswer: """  

    prompt_text = prompt_text6

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt').input_ids
    
    ## GPU warm up
    print("\n")
    print("############# GPU Warm Up ... #############")
    print("\n")
    # time.sleep(5)

    if args.pyramid_enable:
        pyramid_model = get_llama_model(
                args.model_name_or_path,
                torch_dtype=torch.float16,
                device_map="cuda:0",
                cache_dir=args.cache_dir,
                load_in_8bit=True if '70' in args.model_name_or_path or '34' in args.model_name_or_path else False,
            )
        pyramid_model.bfloat16().eval()
        print("Pyramidinfer Model GPU Memory Per GPU (MB): ", f"{torch.cuda.max_memory_allocated(device=pyramid_model.device) / 1024 / 1024:.3f}")
        # pyramid_model = torch.compile(pyramid_model, mode="max-autotune")
        pyramid_model.config.pad_token_id = tokenizer.pad_token_id
        pyramid_config = json.load(open(args.pyramid_config))
        pyramid_model = load_pyramid_config(pyramid_model, pyramid_config)

        # warm up gpu
        input_ids = input_ids.to(pyramid_model.device)
        for i in range(5):
            generate_ids = pyramid_model.generate(input_ids, max_new_tokens=16)

    if args.original_enable:
        from models.modeling_llama import LlamaForCausalLM 
        original_model = LlamaForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.float16,
                device_map="cuda:1",
                cache_dir=args.cache_dir,
                load_in_8bit=True if '70' in args.model_name_or_path or '34' in args.model_name_or_path else False,
            )
        original_model.bfloat16().eval()
        print("Original Model GPU Memory Per GPU:", f"{torch.cuda.max_memory_allocated(device=original_model.device) / 1024 / 1024:.3f}")
        # original_model = torch.compile(original_model, mode="max-autotune")
        original_model.config.pad_token_id = tokenizer.pad_token_id

        # warm up gpu
        input_ids = input_ids.to(original_model.device)
        for i in range(5):
            generate_ids = original_model.generate(input_ids, max_new_tokens=16)

    ## Pyramidinfer start
    if args.pyramid_enable:
        print("\n")
        print("############# Running Pyramidinfer Model #############")
        print("Batch Size: ", args.pyramid_bsz)
        print("Input Tokens Num: ", input_ids.shape[1])
        print("Max New Tokens: ", args.max_new_tokens)
        
        input_ids = tokenizer([prompt_text] * args.pyramid_bsz, add_special_tokens=False, return_tensors='pt').input_ids
        streamer = TextStreamer(tokenizer, skip_prompt=False) if args.pyramid_bsz == 1 else None

        input_ids = input_ids.to(pyramid_model.device)
        torch.cuda.reset_peak_memory_stats(device=pyramid_model.device)
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_time.record()
        generate_ids = pyramid_model.generate(input_ids, streamer=streamer, max_new_tokens=args.max_new_tokens)
        end_time.record()
        torch.cuda.synchronize()
        pyramid_wall_time = start_time.elapsed_time(end_time)
        pyramid_max_gpu_memory = torch.cuda.max_memory_allocated(device=pyramid_model.device)

        all_token_num = generate_ids.shape[1] * generate_ids.shape[0]
        print("\n--------------------------------")
        print(f"Wall Time (ms): {pyramid_wall_time:.3f}")
        print(f"Total Token Num: {all_token_num}")
        print(f"Througthput (token/s): {all_token_num / pyramid_wall_time * 1000:.3f}")
        print(f"Latency (ms/token): {pyramid_wall_time / all_token_num:.3f}")
        print(f"Max GPU Memory Per GPU (MB): {pyramid_max_gpu_memory / 1024 / 1024:.3f}")

    # Orginal start
    if args.original_enable:
        print("\n")
        print("############# Running Original Model #############")
        print("Batch Size: ", args.pyramid_bsz)
        print("Input Tokens Num: ", input_ids.shape[1])
        print("Max New Tokens: ", args.max_new_tokens)
        
        input_ids = tokenizer([prompt_text] * args.original_bsz, add_special_tokens=False, return_tensors='pt').input_ids
        streamer = TextStreamer(tokenizer, skip_prompt=False) if args.original_bsz == 1 else None
        input_ids = input_ids.to(original_model.device) 
        torch.cuda.reset_peak_memory_stats(device=original_model.device)
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_time.record()
        generate_ids = original_model.generate(input_ids, streamer=streamer, max_new_tokens=args.max_new_tokens)
        end_time.record()
        torch.cuda.synchronize()
        original_wall_time = start_time.elapsed_time(end_time)
        original_max_gpu_memory = torch.cuda.max_memory_allocated(device=original_model.device)

        all_token_num = generate_ids.shape[1] * generate_ids.shape[0]
        print("\n--------------------------------")
        print(f"Wall Time (ms): {original_wall_time:.3f}")
        print(f"Total Token Num: {all_token_num}")
        print(f"Througthput (token/s): {all_token_num / original_wall_time * 1000:.3f}")
        print(f"Latency (ms/token): {original_wall_time / all_token_num:.3f}")
        print(f"Max GPU Memory Per GPU (MB): {original_max_gpu_memory / 1024 / 1024:.3f}")

if __name__ == "__main__":
    main()