import base64
import json
import pytest
from openpi.serving.high_level_archive import HighLevelArchive, reconstruct_request

def test_exact_request_reconstruction_and_deduplicated_images(tmp_path):
    archive = HighLevelArchive(tmp_path)
    body = {'model':'gpt-6.2-sol','input':[{'role':'system','content':'Full instructions'},
        {'role':'user','content':[{'type':'input_text','text':'memory'},
        {'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(b'image-bytes').decode(),'detail':'high'}]}]}
    rec = {'epoch':1,'seq':1,'status':'completed','cost_usd':.01}
    directory = archive.prepare(body,rec)
    assert reconstruct_request(tmp_path,directory) == body
    archive.finish(directory,rec,b'{"id":"response"}')
    archive.outcome(directory,{'applied':False})
    archive.prepare(body,{**rec,'seq':2})
    assert len(list((tmp_path/'images').rglob('*.png'))) == 1
    assert json.loads((archive.directory/'index.jsonl').read_text())['seq'] == 1
    image = next((tmp_path/'images').rglob('*.png')); image.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'): reconstruct_request(tmp_path,directory)

def test_archive_io_failure_is_not_silently_ignored(tmp_path):
    blocked=tmp_path/'file';blocked.write_bytes(b'x')
    with pytest.raises(OSError): HighLevelArchive(blocked)
