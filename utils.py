from models.modeling_llama_pyramidinfer import LlamaForCausalLM


def get_llama_model(model_name_or_path, **kwargs):
    model = LlamaForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    return model

def load_pyramid_config(model, config):
    prefill_config = config['prefill_stage']
    for k, v in prefill_config.items():
        setattr(model.config, k, v)
        
    generation_config = config['generation_stage']
    for k, v in generation_config.items():
        setattr(model.config, k, v)
        
    return model