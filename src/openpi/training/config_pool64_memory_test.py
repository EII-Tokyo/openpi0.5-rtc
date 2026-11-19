import ast
from pathlib import Path

def test_literal_pool64_configuration_is_registered_and_keeps_effective_batch():
 source=Path(__file__).with_name('config.py').read_text();tree=ast.parse(source)
 name='eii_twist_cam3_fullft_28repo_h100x2_448_pool256_rtc_nohistory_completedmem_drop20_bs64ga4_20k_20261119'
 node=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and any(k.arg=='name' and isinstance(k.value,ast.Constant) and k.value.value==name for k in n.keywords))
 kw={k.arg:k.value for k in node.keywords};model={k.arg:ast.literal_eval(k.value) for k in kw['model'].keywords}
 assert ast.literal_eval(kw['batch_size'])==64 and ast.literal_eval(kw['gradient_accumulation_steps'])==4
 assert model['image_tokens_per_image']==256 and model['max_token_len']==384 and model['training_time_rtc']
 assert model['image_keys']==('cam_high','cam_left_wrist','cam_right_wrist')
 assert ast.literal_eval(kw['resume']) and not ast.literal_eval(kw['overwrite'])
 assert node.lineno < source[:source.index('_CONFIGS_DICT =')].count('\n')+1
 assert 'max_token_len=384' in ast.get_source_segment(source,kw['data'])

 assert 'bottle_description_dropout_probability=0.2' in ast.get_source_segment(source,kw['data'])
