import json
from unittest.mock import Mock
from openpi.serving.high_level_status import RedisHighLevelStatus, KEY
from openpi.shared.completed_step_memory import CURRENT_TASKS

def test_publishes_decoded_return_and_application_without_credentials():
 c=Mock();c.get.return_value=None;s=RedisHighLevelStatus(c,model='gpt-6.2-sol',checkpoint_step='15000')
 s.event('submitted',{'api_key':'secret','request':'private'})
 record={'seq':1,'epoch':2,'requested_unix':1000.,'camera_timestamps':{'high':999.9},'latency_seconds':4.4,'status':'completed','prediction':{'bottle_description':'green bottle','memory':'','current_task_index':1},'Authorization':'secret'}
 s.event('finished',record,'session/one');s.event('outcome',{'applied':True,'reason':'applied'},'session/one')
 key,raw=c.set.call_args.args;d=json.loads(raw)
 assert key==KEY and d['current_task']==CURRENT_TASKS[0] and d['applied'] is True
 assert d['submitted']==1 and d['completed']==1 and d['inflight']==0
 assert 'secret' not in raw and 'Authorization' not in raw and 'request_interval_seconds' in d

def test_reset_keeps_return_history_without_treating_it_as_current_episode():
 c=Mock();c.get.return_value=json.dumps({'recent_results':[{'id':'old','seq':1,'current_task':'old reply','applied':True}]})
 s=RedisHighLevelStatus(c,model='sol',checkpoint_step=15000);s.event('reset',{})
 d=json.loads(c.set.call_args.args[1]);assert d['status']=='ready' and d['result_from_previous_episode'] is True and d['recent_results'][0]['current_task']=='old reply'

def test_display_failure_does_not_modify_the_inference_contract():
 c=Mock();c.get.return_value=None;c.set.side_effect=OSError('redis unavailable');s=RedisHighLevelStatus(c,model='sol',checkpoint_step=15000);s.event('submitted',{})
 assert s.state['submitted']==1


def test_web_prompt_matches_actual_backend_prompt_and_approved_translation():
 from pathlib import Path
 from openpi_client.runtime.long_memory_high_level import SYSTEM_PROMPT, MEMORY_SCHEMA
 from openpi.shared.sol_eight_task_prompt import PROMPT_VERSION
 c=Mock();c.get.return_value=None;s=RedisHighLevelStatus(c,model='sol',checkpoint_step='15000');s.event('reset',{})
 contract=json.loads(c.set.call_args.args[1])['prompt_contract']
 assert contract['system_prompt_en']==SYSTEM_PROMPT
 assert contract['version']==PROMPT_VERSION
 assert contract['response_schema']==MEMORY_SCHEMA
 assert contract['reply_expiry'] is None
 assert contract['system_prompt_zh']==(Path(__file__).resolve().parents[3]/'docs/sol-eight-task-prompt-zh.txt').read_text().strip()
